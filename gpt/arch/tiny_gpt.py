"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           GPT-2 ARCHITECTURE                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                      │
│  "The cat sat on the"                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TOKENIZER (Byte-Pair Encoding)                                             │
│  "The cat sat on the" → [464, 3797, 3332, 319, 262]                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TOKEN EMBEDDING                          POSITIONAL EMBEDDING              │
│  vocab_size (50257) × d_model (768)       max_seq_len (1024) × d_model      │
│           │                                         │                       │
│           └──────────────┬──────────────────────────┘                       │
│                          ▼                                                  │
│                    Element-wise ADD                                         │
│                          │                                                  │
│                          ▼                                                  │
│              Combined Embeddings [seq_len, 768]                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ╔═══════════════════════════════════════════════════════════════════╗     │
│   ║  TRANSFORMER BLOCK (×12 for GPT-2 Small)                          ║     │
│   ╠═══════════════════════════════════════════════════════════════════╣     │
│   ║                                                                   ║     │
│   ║   Input ──┬──────────────────────────────────────┐                ║     │
│   ║           │                                      │                ║     │
│   ║           ▼                                      │                ║     │
│   ║   ┌───────────────┐                              │                ║     │
│   ║   │  Layer Norm 1 │                              │                ║     │
│   ║   └───────┬───────┘                              │                ║     │
│   ║           │                                      │                ║     │
│   ║           ▼                                      │                ║     │
│   ║   ┌───────────────────────────────────────┐      │                ║     │
│   ║   │  Masked Multi-Head Self-Attention     │      │                ║     │
│   ║   │  (12 heads, d_k = 64 per head)        │      │                ║     │
│   ║   │                                       │      │                ║     │
│   ║   │   Q ─┐                                │      │                ║     │
│   ║   │   K ─┼─► Attention ─► Concat ─► Proj  │      │                ║     │
│   ║   │   V ─┘                                │      │                ║     │
│   ║   └───────────────┬───────────────────────┘      │                ║     │
│   ║                   │                              │                ║     │
│   ║                   ▼                              │                ║     │
│   ║               (+) ADD ◄──────────────────────────┘  Residual 1    ║     │
│   ║                   │                                               ║     │
│   ║           ┌───────┴──────────────────────────────┐                ║     │
│   ║           │                                      │                ║     │
│   ║           ▼                                      │                ║     │
│   ║   ┌───────────────┐                              │                ║     │
│   ║   │  Layer Norm 2 │                              │                ║     │
│   ║   └───────┬───────┘                              │                ║     │
│   ║           │                                      │                ║     │
│   ║           ▼                                      │                ║     │
│   ║   ┌───────────────────────────────────────┐      │                ║     │
│   ║   │  Feed-Forward Network                 │      │                ║     │
│   ║   │                                       │      │                ║     │
│   ║   │  Linear(768 → 3072)                   │      │                ║     │
│   ║   │       │                               │      │                ║     │
│   ║   │       ▼                               │      │                ║     │
│   ║   │     GELU                              │      │                ║     │
│   ║   │       │                               │      │                ║     │
│   ║   │       ▼                               │      │                ║     │
│   ║   │  Linear(3072 → 768)                   │      │                ║     │
│   ║   └───────────────┬───────────────────────┘      │                ║     │
│   ║                   │                              │                ║     │
│   ║                   ▼                              │                ║     │
│   ║               (+) ADD ◄──────────────────────────┘  Residual 2    ║     │
│   ║                   │                                               ║     │
│   ║                   ▼                                               ║     │
│   ║               Output                                              ║     │
│   ║                                                                   ║     │
│   ╚═══════════════════════════════════════════════════════════════════╝     │
│                                      │                                      │
│                              (Repeat ×12)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINAL LAYER NORM                                                           │
│  LayerNorm(768)                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT PROJECTION (Language Model Head)                                    │
│  Linear(768 → 50257)  [Weight tied with token embeddings]                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SOFTMAX                                                                    │
│  Probability distribution over vocabulary                                   │
│  P("mat" | "The cat sat on the") = 0.12                                     │
│  P("floor" | "The cat sat on the") = 0.08                                   │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                     │
│  Next token prediction: "mat" (or sample from distribution)                 │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import math
import pathlib
import sys
import tiktoken

import numpy as np
import requests as r
from tinygrad import Tensor, nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


class TinyEncoder:

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __call__(self, text: str) -> Tensor:
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        token_ids = Tensor(token_ids)
        return token_ids

    def decode(self, idx: Tensor) -> str:
        return self.tokenizer.decode()


class MultiHeadAttention:

    def __init__(self, d_in: int, d_out: int, context_length: int, n_heads: int, dropout: float = 0.0, bias=False) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.n_heads = n_heads
        self.head_dim = self.d_out // n_heads

        assert self.d_out % n_heads == 0

        # Linear Projections
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)

        self.dropout = dropout
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)

        self.k_cache = None
        self.v_cache = None

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def __call__(self, x: Tensor, use_cache: bool = False) -> Tensor:
        b, num_tokens, _ = x.shape

        # linear projections (B, num_tokens, d_out)
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        # split heads (B, num_tokens, d_out) -> (B, n_heads, num_tokens, head_dim)
        keys = keys.reshape(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        queries = queries.reshape(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask example (chunked input: tokens 3,4,5 with tokens 0-2 cached):
        #          k0   k1   k2   k3   k4   k5
        #    q3 [[ 0.2  0.5  0.1  0.8 -inf -inf ]   ← attends to 0-3
        #    q4  [ 0.1  0.3  0.6  0.2  0.7 -inf ]   ← attends to 0-4
        #    q5  [ 0.4  0.2  0.5  0.3  0.1  0.6 ]]  ← attends to 0-5
        if use_cache:
            if self.k_cache is not None:
                self.k_cache = self.k_cache.cat(keys, dim=2).realize()
                self.v_cache = self.v_cache.cat(values, dim=2).realize()
                keys = self.k_cache
                values = self.v_cache
            else:
                self.k_cache = keys.realize()
                self.v_cache = values.realize()

        attention_scores = queries @ keys.transpose(2, 3)

        if use_cache:
            num_tokens_k, num_tokens_q = keys.shape[2], queries.shape[2]
            causal_mask = Tensor.triu(Tensor.ones(num_tokens_q, num_tokens_k), diagonal=num_tokens_k - num_tokens_q + 1).bool()
        else:
            causal_mask = Tensor.triu(Tensor.ones(num_tokens, num_tokens), diagonal=1).bool()

        attention_scores = attention_scores.masked_fill(causal_mask, -math.inf)
        attention_weights = (attention_scores / self.head_dim ** 0.5).softmax(axis=-1)
        attention_weights = attention_weights.dropout(self.dropout)

        context_vec = attention_weights @ values

        # (B, n_heads, num_tokens, head_dim) -> (B, num_tokens, d_out)
        # context_vec = context_vec.view(b, num_tokens, self.d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class GELU:

    def __call__(self, x: Tensor) -> Tensor:
        result = x + 0.044715 * x.pow(3)
        result = (2 / math.pi) ** 0.5 * result
        result = 1 + result.tanh()
        result = 0.5 * x * result
        return result


class FeedForward:

    def __init__(self, emb_dim: int):
        self.emb_dim = emb_dim
        self.expansion = nn.Linear(emb_dim, emb_dim * 4, bias=True)    # Expansion
        self.activation = GELU()                                         # Activation
        self.projection = nn.Linear(4 * emb_dim, emb_dim, bias=True)    # Projection (reduction)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.expansion(x)
        x = self.activation(x)
        x = self.projection(x)
        return x


class LayerNorm:
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5                                           # Epsilon
        self.scale = Tensor.ones(emb_dim, requires_grad=True)     # Gamma
        self.shift = Tensor.zeros(emb_dim, requires_grad=True)    # Beta

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, correction=False)
        norm_x = (x - mean) / (var + self.eps).sqrt()
        return self.scale * norm_x + self.shift


class TransformerBlock:

    def __init__(self, cfg: dict[str, int]):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            n_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            bias=cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg['emb_dim'])
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_rate = cfg['drop_rate']  # applied at multiple levels to prevent overfitting at each level

    def __call__(self, x: Tensor, use_cache: bool = False) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, use_cache=use_cache)
        x = x.dropout(self.drop_rate)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x.dropout(self.drop_rate)
        x = x + shortcut

        return x


class TinyGPTModel:
    def __init__(self, cfg: dict[str, object]):

        # Embedding
        self.emb_layer = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_layer = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_rate = cfg["drop_rate"]

        # Transformer blocks
        self.trf_blocks = [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]

        # Layer norms
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=True)

        self.current_pos = 0

    def __call__(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        num_batches, seq_len = in_idx.shape
        embedding = self.emb_layer(in_idx)

        if use_cache:
            positions = Tensor.arange(self.current_pos, self.current_pos + seq_len)
            self.current_pos += seq_len
        else:
            positions = Tensor.arange(seq_len)
            self.current_pos = seq_len

        pos_embeddings = self.pos_layer(positions)

        x = embedding + pos_embeddings
        x = x.dropout(self.drop_rate)
        for block in self.trf_blocks:
            x = block(x, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_cache(self) -> None:
        self.current_pos = 0
        for blk in self.trf_blocks:
            blk.attention.reset_cache()


def download_gpt_model(base_url, model_size, download_dir):
    models_dir = pathlib.Path(download_dir)
    models_dir.mkdir(exist_ok=True)

    files = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    for filename in files:
        url = f"{base_url}/{model_size}/{filename}"
        filepath = models_dir.joinpath(filename)
        if not filepath.exists():
            print(f"Downloading {filename}...")
            response = r.get(url)
            with open(filepath, "wb") as f:
                f.write(response.content)


def load_tf_checkpoint(models_dir):
    import tensorflow as tf

    models_dir = pathlib.Path(models_dir)
    tf_ckpt_path = models_dir.joinpath("model.ckpt")
    params = {}

    for name, _ in tf.train.list_variables(tf_ckpt_path):
        arr = tf.train.load_variable(tf_ckpt_path, name)
        params[name] = arr
    return params


def to_tensor(arr: np.array) -> Tensor:
    return Tensor(arr.astype(np.float32))


def assign(left: Tensor, right: Tensor):
    if left.shape != right.shape:
        raise ValueError(f'Dimensions mismatch: {left.shape} != {right.shape}')
    return right


def load_model(params) -> TinyGPTModel:
    tiny_model = TinyGPTModel(GPT_CONFIG_124M)

    tiny_model.emb_layer.weight = assign(tiny_model.emb_layer.weight, to_tensor(params["model/wte"]))
    tiny_model.pos_layer.weight = assign(tiny_model.pos_layer.weight, to_tensor(params["model/wpe"]))

    for i, block in enumerate(tiny_model.trf_blocks):
        # Layer norms
        block.norm1.scale = assign(block.norm1.scale, to_tensor(params[f"model/h{i}/ln_1/g"]))
        block.norm1.shift = assign(block.norm1.shift, to_tensor(params[f"model/h{i}/ln_1/b"]))
        block.norm2.scale = assign(block.norm2.scale, to_tensor(params[f"model/h{i}/ln_2/g"]))
        block.norm2.shift = assign(block.norm2.shift, to_tensor(params[f"model/h{i}/ln_2/b"]))

        # Attention - weights
        qkv_w = params[f"model/h{i}/attn/c_attn/w"][0]  # Shape: (768, 2304)
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=-1)

        block.attention.W_q.weight = assign(block.attention.W_q.weight, to_tensor(q_w.T))
        block.attention.W_k.weight = assign(block.attention.W_k.weight, to_tensor(k_w.T))
        block.attention.W_v.weight = assign(block.attention.W_v.weight, to_tensor(v_w.T))

        # Attention - bias
        qkv_b = params[f"model/h{i}/attn/c_attn/b"]
        q_b, k_b, v_b = np.split(qkv_b, 3, axis=-1)

        block.attention.W_q.bias = assign(block.attention.W_q.bias, to_tensor(q_b))
        block.attention.W_k.bias = assign(block.attention.W_k.bias, to_tensor(k_b))
        block.attention.W_v.bias = assign(block.attention.W_v.bias, to_tensor(v_b))

        # Attention = Output projection
        block.attention.out_proj.weight = assign(block.attention.out_proj.weight, to_tensor(params[f"model/h{i}/attn/c_proj/w"][0].T))
        block.attention.out_proj.bias = assign(block.attention.out_proj.bias, to_tensor(params[f"model/h{i}/attn/c_proj/b"]))

        # FFN
        block.ff.expansion.weight = assign(block.ff.expansion.weight, to_tensor(params[f"model/h{i}/mlp/c_fc/w"][0].T))
        block.ff.expansion.bias = assign(block.ff.expansion.bias, to_tensor(params[f"model/h{i}/mlp/c_fc/b"]))
        block.ff.projection.weight = assign(block.ff.projection.weight, to_tensor(params[f"model/h{i}/mlp/c_proj/w"][0].T))
        block.ff.projection.bias = assign(block.ff.projection.bias, to_tensor(params[f"model/h{i}/mlp/c_proj/b"]))

    # Final norm
    tiny_model.final_norm.scale = to_tensor(params["model/ln_f/g"])
    tiny_model.final_norm.shift = to_tensor(params["model/ln_f/b"])

    # Output head
    tiny_model.out_head.weight = to_tensor(params["model/wte"])
    return tiny_model


def sample_top_p(logits: Tensor, top_p: float = 0.9) -> Tensor:
    # logits: (B, V) -> (1, 50257)
    # Convert logits to probabilities
    probs = logits.softmax(axis=-1)

    # Limit to top 100 tokens
    sorted_probs, topk_ids = probs.topk(50, dim=-1)

    # Compute the cumulative sum
    cum = sorted_probs.cumsum(axis=-1)

    # Mask for elements whose cum sum is less or equal to p
    keep = ((cum - sorted_probs) < top_p).float()

    # Ensure at least 1 token kept
    keep = Tensor.ones((logits.shape[0], 1), dtype=sorted_probs.dtype, device=sorted_probs.device).cat(keep[:, 1:], dim=-1)

    # Mask + renormalize
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum(axis=-1, keepdim=True)

    # Sample in sorted space, then map back to vocab ids
    sampled_pos = filtered.multinomial()
    sampled_pos.realize()

    token = topk_ids.gather(-1, sampled_pos)
    return token.realize()


if __name__ == "__main__":
    print('Downloading model...')
    download_gpt_model(
        base_url="https://openaipublic.blob.core.windows.net/gpt-2/models",
        model_size="124M",
        download_dir="gpt2_weights"
    )

    print("Load tensorflow checkpoint...")
    nn.Embedding(1, 1)  # WTF?!

    params = load_tf_checkpoint("gpt2_weights")

    print("Loading model...")
    tiny_model = load_model(params)
    tiny_model.reset_cache()
    print('Model loaded.')

    encoder = TinyEncoder()
    input_batch = encoder("What is the purpose of life?").unsqueeze(0)
    logits = tiny_model(input_batch, use_cache=True)
    max_new_tokens = 30
    idx = input_batch
    last_line_count = 0

    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]

        probas = logits.softmax(axis=-1)
        nxt = probas.argmax(axis=-1, keepdim=True).realize()

        nxt = sample_top_p(logits)

        idx = idx.cat(nxt, dim=1)
        logits = tiny_model(nxt, use_cache=True).realize()

        output = encoder.tokenizer.decode(idx.squeeze(0).tolist())

        # Clear previous output
        if last_line_count > 0:
            sys.stdout.write(f"\033[{last_line_count}A")
            sys.stdout.write("\033[J")

        print(output, end="\r")
        last_line_count = output.count('\n')
