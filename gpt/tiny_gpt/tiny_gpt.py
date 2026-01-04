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
import heapq
import math
import pathlib
import random
import typing
import tiktoken

import requests as r  # type: ignore
from tinygrad import Tensor, Device, nn  # type: ignore
from tinygrad.nn.state import torch_load  # type: ignore


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_350M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": True
}


class TinyEncoder:

    def __init__(self) -> None:
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __call__(self, text: str) -> Tensor:
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        token_ids = Tensor(token_ids)
        return token_ids


class MultiHeadAttention:

    def __init__(self, d_in: int, d_out: int, context_length: int, n_heads: int, dropout: float = 0.0, bias: bool = False) -> None:
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

    def __init__(self, cfg: dict[str, int | bool | float]):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=int(cfg['emb_dim']),
            d_out=int(cfg['emb_dim']),
            context_length=int(cfg['context_length']),
            n_heads=int(cfg['n_heads']),
            dropout=cfg['drop_rate'],
            bias=bool(cfg['qkv_bias'])
        )

        self.ff = FeedForward(int(cfg['emb_dim']))
        self.norm1 = LayerNorm(int(cfg['emb_dim']))
        self.norm2 = LayerNorm(int(cfg['emb_dim']))
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
    def __init__(self, cfg: dict[str, int | bool | float]):
        # Embedding
        self.emb_layer = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_layer = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_rate = cfg["drop_rate"]

        # Transformer blocks
        self.trf_blocks = [TransformerBlock(cfg) for _ in range(int(cfg["n_layers"]))]

        # Layer norms
        self.final_norm = LayerNorm(int(cfg["emb_dim"]))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=True)

        self.current_pos = 0

    def embed(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        """Token + positional embeddings"""
        _, seq_len = in_idx.shape
        embedding = self.emb_layer(in_idx).realize()

        if use_cache:
            positions = Tensor.arange(self.current_pos, self.current_pos + seq_len)
            self.current_pos += seq_len
        else:
            positions = Tensor.arange(seq_len)
            self.current_pos = seq_len

        pos_embeddings = self.pos_layer(positions)
        return embedding + pos_embeddings

    def __call__(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        """Forward"""
        x = self.embed(in_idx, use_cache)
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

    def forward_layers(self, x: Tensor, use_cache: bool = False) -> typing.Generator[tuple[int, Tensor], None, None]:
        """Generator yielding hidden state after each layer."""
        x = x.dropout(self.drop_rate)
        for i, block in enumerate(self.trf_blocks):
            x = block(x, use_cache=use_cache)
            yield i, x


def download_gpt_model(model_size: str) -> pathlib.Path:
    url = f"https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin"
    out_file = pathlib.Path(f'pytorch_model_{model_size}.bin')

    if out_file.exists():
        return out_file

    print(f"Downloading {out_file.name}...")
    response = r.get(url)
    with open(out_file, "wb") as f:
        f.write(response.content)
    return out_file


def assign(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError(f'Dimensions mismatch: {left.shape} != {right.shape}')
    return right.contiguous().realize()


def load_model(weights: dict[str, Tensor], model_size: str) -> TinyGPTModel:
    if model_size == 'gpt2':
        cfg = GPT_CONFIG_124M
    elif model_size == 'gpt2-medium':
        cfg = GPT_CONFIG_350M
    elif model_size == 'gpt2-large':
        cfg = GPT_CONFIG_774M
    else:
        raise NotImplementedError(model_size)

    tiny_model = TinyGPTModel(cfg)

    tiny_model.emb_layer.weight = assign(tiny_model.emb_layer.weight, weights["wte.weight"])
    tiny_model.pos_layer.weight = assign(tiny_model.pos_layer.weight, weights["wpe.weight"])

    for i, block in enumerate(tiny_model.trf_blocks):
        # Layer norms
        block.norm1.scale = assign(block.norm1.scale, weights[f"h.{i}.ln_1.weight"])
        block.norm1.shift = assign(block.norm1.shift, weights[f"h.{i}.ln_1.bias"])
        block.norm2.scale = assign(block.norm2.scale, weights[f"h.{i}.ln_2.weight"])
        block.norm2.shift = assign(block.norm2.shift, weights[f"h.{i}.ln_2.bias"])

        # Attention - weights
        qkv_w = weights[f"h.{i}.attn.c_attn.weight"]  # Shape: (768, 2304)
        q_w, k_w, v_w = qkv_w.chunk(3, dim=-1)

        block.attention.W_q.weight = assign(block.attention.W_q.weight, q_w.T)
        block.attention.W_k.weight = assign(block.attention.W_k.weight, k_w.T)
        block.attention.W_v.weight = assign(block.attention.W_v.weight, v_w.T)

        # Attention - bias
        qkv_b = weights[f"h.{i}.attn.c_attn.bias"]
        q_b, k_b, v_b = qkv_b.chunk(3, dim=-1)

        block.attention.W_q.bias = assign(block.attention.W_q.bias, q_b)
        block.attention.W_k.bias = assign(block.attention.W_k.bias, k_b)
        block.attention.W_v.bias = assign(block.attention.W_v.bias, v_b)

        # Attention = Output projection
        block.attention.out_proj.weight = assign(block.attention.out_proj.weight, weights[f"h.{i}.attn.c_proj.weight"].T)
        block.attention.out_proj.bias = assign(block.attention.out_proj.bias, weights[f"h.{i}.attn.c_proj.bias"])

        # FFN
        block.ff.expansion.weight = assign(block.ff.expansion.weight, weights[f"h.{i}.mlp.c_fc.weight"].T)
        block.ff.expansion.bias = assign(block.ff.expansion.bias, weights[f"h.{i}.mlp.c_fc.bias"])
        block.ff.projection.weight = assign(block.ff.projection.weight, weights[f"h.{i}.mlp.c_proj.weight"].T)
        block.ff.projection.bias = assign(block.ff.projection.bias, weights[f"h.{i}.mlp.c_proj.bias"])

    # Final norm
    tiny_model.final_norm.scale = assign(tiny_model.final_norm.scale, weights["ln_f.weight"])
    tiny_model.final_norm.shift = assign(tiny_model.final_norm.shift, weights["ln_f.bias"])

    # Output head
    tiny_model.out_head.weight = assign(tiny_model.out_head.weight, weights["wte.weight"])
    return tiny_model


def sample_top_p(logits_batch: Tensor, top_p: float = 0.9) -> Tensor:
    """Nucleus sampling
    Done in Python, because I do not know how to do this in tinygrad."""
    logits_list = logits_batch.realize().tolist()
    out = []

    for logits in logits_list:
        max_logit = max(logits)

        # softmax (stable)
        probs = [math.exp(logit - max_logit) for logit in logits]
        s = sum(probs)
        probs = [p / s for p in probs]

        # sort by probability
        pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        # top-p truncation (only include values with cumsum up to p)
        cum = 0.0
        filtered = []
        for idx, p in pairs:
            cum += p
            filtered.append((idx, p))
            if cum >= top_p:
                break

        indices, weights = zip(*filtered)
        token = random.choices(indices, weights=weights, k=1)[0]
        out.append(token)

    return Tensor(out, device=logits_batch.device).reshape(logits_batch.shape[0], 1)


def apply_repetition_penalty(logits: Tensor, generated_ids: Tensor, penalty: float = 1.1) -> Tensor:
    """Penalize tokens that have already been generated."""
    generated_list = generated_ids.squeeze(0).tolist()
    logits_list = logits.squeeze(0).tolist()

    for token_id in set(generated_list):
        if logits_list[token_id] > 0:
            logits_list[token_id] /= penalty
        else:
            logits_list[token_id] *= penalty

    return Tensor(logits_list).unsqueeze(0)


def prevent_ngram_repetition(logits: Tensor, generated_ids: Tensor, n: int = 3) -> Tensor:
    """Prevent n-gram repetition."""
    generated_list = generated_ids.squeeze(0).tolist()
    logits_list = logits.squeeze(0).tolist()

    if len(generated_list) < n:
        return logits

    context = tuple(generated_list[-(n - 1):])

    banned_ids = set()
    for i in range(len(generated_list) - n + 1):
        if tuple(generated_list[i:i + n - 1]) == context:
            # Last n-1 tokens match [A, B] <=> [A, B]
            # Therefore, bann the next token, because this token was already generated
            # (e.g. C, if generated_list contains [A, B, C])
            banned_ids.add(generated_list[i + n - 1])

    for x in banned_ids:
        logits_list[x] = float('-inf')
    return Tensor(logits_list).unsqueeze(0)


def explore_hidden_layers(model: TinyGPTModel, text: str, max_new_tokens: int, encoder: TinyEncoder) -> None:
    """Expore hidden states layer by layer."""
    idx = encoder(text).unsqueeze(0)
    logits = model(idx, use_cache=False)
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]
        nxt = logits.argmax(axis=-1, keepdim=True)
        idx = idx.cat(nxt, dim=1)

        print('------')
        print(encoder.tokenizer.decode(idx.squeeze(0).tolist()))

        x = model.embed(idx, False)
        for i, layer in model.forward_layers(x, False):
            normed = model.final_norm(layer)
            logits = model.out_head(normed)
            logits_list = logits[:, -1, :].squeeze().tolist()
            top = heapq.nlargest(5, range(len(logits_list)), key=lambda i: logits_list[i])
            print(i, [tiktoken.get_encoding("gpt2").decode([t,]) for t in top])


def load_weights(fp: pathlib.Path) -> dict[str, Tensor]:
    weights = torch_load(fp)
    weights = {k: v.to(Device.DEFAULT).realize() for k, v in weights.items()}
    return weights
