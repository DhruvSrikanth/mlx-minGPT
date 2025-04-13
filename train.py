import uuid
import os
import argparse
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
import wandb
from rich.progress import Progress
from rich import print as rprint


# ----------- Tokenizer -----------
class Tokenizer:
    def __init__(self, raw_text: str):
        # Build vocabulary
        vocab = sorted(list(set(raw_text)))
        # Build tokenizer
        self.vocab_size = len(vocab)
        self.itos = {i: c for i, c in enumerate(vocab)}  # int to string
        self.stoi = {c: i for i, c in enumerate(vocab)}  # string to int

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])


# ----------- Dataset -----------
def read_dataset(data_path: str) -> list[int]:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text


def split_data(raw_text: str, tokenizer: Tokenizer, train_val_split: float, ctx_len: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    data = tokenizer.encode(raw_text)
    split = int(train_val_split * len(data))
    train_data = data[:split]
    val_data = data[split:]
    X_train = mx.array([train_data[i:i + ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)])
    y_train = mx.array([train_data[i + 1:i + ctx_len + 1] for i in range(0, len(train_data) - ctx_len, ctx_len)])
    X_val = mx.array([val_data[i:i + ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)])
    y_val = mx.array([val_data[i + 1:i + ctx_len + 1] for i in range(0, len(val_data) - ctx_len, ctx_len)])
    return X_train, y_train, X_val, y_val


def get_batches(X: mx.array, y: mx.array, batch_size: int, shuffle: bool = False):
    # Shuffle data if required
    if shuffle:
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        ids = mx.array(ids)
        X = X[ids]
        y = y[ids]

    # Yield batches
    for i in range(0, X.shape[0], batch_size):
        input = X[i:i + batch_size]
        label = y[i:i + batch_size]
        yield input, label


# ----------- Models Components -----------
class MLP(nn.Module):
    def __init__(self, n_embed: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, bias: bool, dropout: float, n_heads: int, ctx_len: int):
        super().__init__()
        assert n_embed % n_heads == 0, f"n_embed ({n_embed}) must be divisible by n_heads ({n_heads})"
        # kqv projections for all heads
        self.c_kqv = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = n_embed // self.n_heads
        self.sqrt_d = math.sqrt(self.head_dim)
        # Causal mask for all heads (1, 1, ctx_len, ctx_len)
        self.causal_mask = mx.tril(x=mx.ones((ctx_len, ctx_len)), k=0)[None, None, :, :]

    def __call__(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        # Project x to K, Q, V for all heads
        K, Q, V = mx.split(self.c_kqv(x), indices_or_sections=3, axis=2)
        # B, T, C -> B, T, n_heads, head_dim -> B, n_heads, T, head_dim
        K = mx.as_strided(K, (B, T, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        Q = mx.as_strided(Q, (B, T, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        V = mx.as_strided(V, (B, T, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        # Attention = softmax((Q K^T / sqrt(d)) with mask)
        # B, n_heads, T, head_dim @ B, n_heads, head_dim, T -> B, n_heads, T, T
        A = ((Q @ K.transpose(0, 1, 3, 2)) / self.sqrt_d)
        # Apply mask
        A = mx.where(self.causal_mask[:, :, :T, :T] == 0, -np.inf, A)
        A_norm = mx.softmax(A, axis=-1)
        A_norm = self.attn_dropout(A_norm)
        # B, n_heads, T, T @ B, n_heads, T, head_dim -> B, n_heads, T, head_dim
        y = A_norm @ V
        # B, n_heads, T, head_dim -> B, T, n_heads, head_dim -> B, T, C (reassemble heads)
        y = mx.contiguous(y.transpose(0, 2, 1, 3)).reshape((B, T, C))
        # Project y back to output space
        y = self.c_proj(self.resid_dropout(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embed: int, bias: bool, dropout: float, n_heads: int, ctx_len: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(dims=n_embed, bias=bias)
        self.attention = MultiHeadAttention(n_embed=n_embed, bias=bias, dropout=dropout, n_heads=n_heads, ctx_len=ctx_len)
        self.norm_2 = nn.LayerNorm(dims=n_embed, bias=bias)
        self.mlp = MLP(n_embed=n_embed, dropout=dropout, bias=bias)

    def __call__(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, bias: bool, dropout: float, n_heads: int, n_layers: int, ctx_len: int):
        super().__init__()
        self.ctx_len = ctx_len
        # Token and position embeddings
        # vocab_size -> n_embed
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(self.ctx_len, n_embed)
        # Embedding dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embed=n_embed, bias=bias, dropout=dropout, n_heads=n_heads, ctx_len=self.ctx_len)
            for _ in range(n_layers)
        ])
        # n_embed -> vocab_size
        self.ln_f = nn.LayerNorm(dims=n_embed, bias=bias)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=bias)
        self.wte.weight = self.lm_head.weight  # share weights between wte and lm_head since they learn similar representations
        self._init_params()

    def _init_params(self):
        normal_init = nn.init.normal(mean=0.0, std=0.02)
        new_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.layers.linear.Linear):
                new_params.append((name + '.weight', normal_init(module.weight)))
                if 'bias' in module:
                    new_params.append((name + '.bias', mx.zeros(module.bias.shape)))
            elif isinstance(module, nn.layers.embedding.Embedding):
                new_params.append((name + '.weight', normal_init(module.weight)))
        self = self.update(utils.tree_unflatten(new_params))

    def __call__(self, x: mx.array) -> mx.array:
        T = x.shape[1]
        assert T <= self.ctx_len, f"Sequence length {T} exceeds context length {self.ctx_len}"
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(start=0, stop=T, step=1, dtype=mx.int32))
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, ctx: mx.array, max_new_tokens: int, temperature: float = 1.0) -> mx.array:
        # Set the eval mode
        self.train(False)
        for _ in range(max_new_tokens):
            # Crop context to most recent ctx_len tokens
            lm_ctx = ctx[:, -self.ctx_len:]
            # Get the probabilities of the next token
            # B, T, C -> B, -1, C
            logits = self(lm_ctx)[:, -1, :]
            # Unnormalized probabilities adjusted by temperature
            probs = logits / temperature
            # Sample next token and add to context
            next_tok = mx.random.categorical(probs, num_samples=1)
            ctx = mx.concatenate((ctx, next_tok), axis=1)
        return ctx


# ----------- Loss Function -----------
def loss_fn(m: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
    # Forward pass
    logits = m(xb)
    B, T, C = logits.shape
    # Compute loss
    loss = nn.losses.cross_entropy(logits.reshape(B * T, C), yb.reshape(B * T), reduction='mean')
    return loss


# ----------- Training -----------
def train(X_train: mx.array, y_train: mx.array, X_val: mx.array, y_val: mx.array, batch_size: int, model: nn.Module, lr: float, betas: tuple[float, float], weight_decay: float, max_grad_norm: float, steps: int, eval_frequency: int, tokenizer: Tokenizer, save_dir: str, log: bool) -> nn.Module:
    # Since MLX is lazy, we need to evaluate the model to set the initial state
    mx.eval(model.parameters())
    # Since MLX is pure, we need to define the forward and backward pass as a function
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # Build optimizer
    optimizer = optim.AdamW(learning_rate=lr, betas=betas, weight_decay=weight_decay)

    train_data_iterator = iter(get_batches(X=X_train, y=y_train, batch_size=batch_size, shuffle=True))

    # Training loop
    with Progress() as pbar:
        task = pbar.add_task("[cyan] Training...", total=steps)
        for step in range(steps):
            # Get the data to train
            try:
                xb, yb = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(get_batches(X=X_train, y=y_train, batch_size=batch_size, shuffle=True))
                xb, yb = next(train_data_iterator)

            # -------- Training --------
            # Set model to training mode
            model = model.train(True)
            # Forward pass + backward pass
            loss, grads = loss_and_grad(model, xb, yb)
            # Perform gradient clipping
            clipped_grads, _ = optim.clip_grad_norm(grads=grads, max_norm=max_grad_norm)
            # Update parameters and optimizer state
            optimizer.update(model, clipped_grads)
            # Since MLX is lazy, we need to evaluate the previous line's operations
            mx.eval(model.parameters(), optimizer.state)
            train_loss = loss.item()

            if log:
                wandb.log({"train_loss": train_loss}, step=step)

            # -------- Validation --------
            # Evaluate every eval_frequency steps and at the last step as well
            val_loss = 0.0
            if step % eval_frequency == 0 or step == steps - 1:
                # Set model to evaluation mode
                model = model.train(False)
                val_loss = 0.0
                for xb, yb in get_batches(X=X_val, y=y_val, batch_size=batch_size, shuffle=False):
                    # Forward pass
                    val_loss += loss_fn(m=model, xb=xb, yb=yb)
                    # Since MLX is lazy, we need to evaluate the previous line's operations
                    mx.eval(val_loss)

                # Compute average validation loss over all batches
                val_loss = val_loss.item() / (math.ceil(len(X_val) / batch_size))

                if log:
                    wandb.log({"val_loss": val_loss}, step=step)

                    # Save model
                    step_dir = os.path.join(save_dir, str(step))
                    os.makedirs(step_dir, exist_ok=True)
                    model.save_weights(os.path.join(step_dir, 'model.safetensors'))

                    # Generate and save completion
                    completion = generate_completion(start="To be or not to be,", model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=1.0)
                    with open(os.path.join(step_dir, 'completion.txt'), 'w', encoding='utf-8') as f:
                        f.write(completion)

            pbar.update(task, advance=1, description=f"[cyan]Step {step}/{steps} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model


# ----------- Inference -----------
def generate_completion(start: str, model: nn.Module, tokenizer: Tokenizer, max_new_tokens: int, temperature: float = 1.0) -> str:
    # Shape (1, len(tokenizer.encode(start))) -> B, T
    start_ctx = mx.array([tokenizer.encode(start)])
    completion_tokens = model.generate(ctx=start_ctx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
    completion = tokenizer.decode(completion_tokens)
    return completion


# ----------- Main -----------
def main(args: argparse.Namespace):
    assert args.steps > args.eval_frequency, f"steps ({args.steps}) must be greater than eval_frequency ({args.eval_frequency})"

    # Create save directory if it doesn't exist with uuid
    save_dir = os.path.join(args.save_dir, str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    # Initialize wandb
    if args.log:
        wandb.init(
            project="mlx-gpt",  # replace with your project name
            name=f"run-{save_dir.split('/')[-1]}",
            config=vars(args)
        )

    # Set random seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Set the default device to the gpu
    mx.set_default_device(mx.gpu)

    # ----------- Build dataset -----------
    # Read, tokenize and split
    raw_text = read_dataset(data_path=args.data_path)
    tokenizer = Tokenizer(raw_text=raw_text)
    X_train, y_train, X_val, y_val = split_data(raw_text=raw_text, tokenizer=tokenizer, train_val_split=args.train_val_split, ctx_len=args.ctx_len)

    # ----------- Build model -----------
    model = GPT(vocab_size=tokenizer.vocab_size, n_embed=args.n_embed, bias=args.bias, dropout=args.dropout, n_heads=args.n_heads, n_layers=args.n_layers, ctx_len=args.ctx_len)

    # Print Summary
    rprint(f"Log dir:[blue underline]{save_dir}[/blue underline]")
    rprint(f"Number of parameters: [yellow]{(sum(v.size for _, v in utils.tree_flatten(model.parameters())) / 1e6):.2f}M[/yellow]")

    # ----------- Train model -----------
    trained_model = train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=args.batch_size, model=model, lr=args.lr, betas=args.betas, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, steps=args.steps, eval_frequency=args.eval_frequency, tokenizer=tokenizer, save_dir=save_dir, log=args.log)

    # ----------- Generate completion -----------
    print(generate_completion(start="To be or not to be,", model=trained_model, tokenizer=tokenizer, max_new_tokens=1000, temperature=1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    # ----------- Data Hyperparameters -----------
    parser.add_argument('--data_path', type=str, default='dataset.txt')
    parser.add_argument('--train_val_split', type=float, default=0.9)

    # ----------- Model Hyperparameters -----------
    parser.add_argument('--ctx_len', type=int, default=8)
    parser.add_argument('--n_embed', type=int, default=384)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)

    # ----------- Training Hyperparameters -----------
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--betas', type=tuple[float, float], default=(0.9, 0.95))
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_frequency', type=int, default=250)

    # ----------- Logging Parameters -----------
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='.runs/')

    # ----------- Run -----------
    main(args=parser.parse_args())
