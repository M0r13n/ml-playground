import math
from tinygrad import Tensor, nn, UOp  # type: ignore[attr-defined]
from tinygrad.nn.state import gguf_load
from tinygrad.apps.llm import SimpleTokenizer

QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
}


class RMSNorm:
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        self.scale = Tensor.ones(emb_dim, requires_grad=True)  # scale but no shift
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(axis=-1, keepdim=True)
        return x * (variance + self.eps).rsqrt() * self.scale


class SiLU:

    def __call__(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class FeedForward:

    def __init__(self, emb_dim: int, hidden_dim: int):
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = SiLU()

    def __call__(self, x: Tensor) -> Tensor:
        # x_1 = self.gate_proj(x)     # (B, num_tokens, hidden_dim)
        # x_2 = self.up_proj(x)       # (B, num_tokens, hidden_dim)
        # x = self.silu(x_1) * x_2    # (B, num_tokens, hidden_dim)
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


def precompute_rope_params(head_dim: int, theta_base: int | float = 10_000, context_length: int = 4096) -> tuple[Tensor, Tensor]:
    """https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb"""
    if not head_dim % 2 == 0:
        raise ValueError("head_dim must be even")

    inv_freq = 1.0 / (theta_base ** (Tensor.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    positions = Tensor.arange(context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = angles.cat(angles, dim=1)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def compute_rope(x: Tensor, cos: Tensor, sin: Tensor, position_offset: int | UOp = 0) -> Tensor:
    """https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb"""
    _, _, seq_len, head_dim = x.shape
    if not head_dim % 2 == 0:
        raise ValueError("head_dim must be even")

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[position_offset:position_offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[position_offset:position_offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = (x2 * -1).cat(x1, dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated


class GroupedQueryAttention:
    """Similar to MultiHeadAttention of gpt2 with two significant changes:

    1. uses ROPE positional embeddings
    2. shares key and value projections among multiple attention heads (less parameters required)

    MHA

    q1  q2  q3  q4
    k1  k2  k3  k4
    v1  v2  v3  v4

    GQA

    q1  q2  q3  q4           q1  q2  q3  q4
      k1      k2      <=>    k1  k1  k2  k2
      v1      v2             v1  v1  v2  v2
    """

    def __init__(self, d_in: int, n_heads: int, num_kv_groups: int, head_dim: int) -> None:
        self.d_in = d_in
        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_out = n_heads * head_dim
        self.group_size = n_heads // num_kv_groups

        # Linear Projections (Q, K, V) as a single matrix
        self.attn_q = nn.Linear(d_in, n_heads * head_dim, bias=False)        # Q has all heads
        self.attn_k = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)  # K has fewer heads
        self.attn_v = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)  # V has fewer heads

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False)

        # KV-Cache
        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, use_cache: bool = False) -> Tensor:
        b, num_tokens, _ = x.shape

        # Queries, Keys and Values
        q = self.attn_q(x)  # (B, num_tokens, d_out)
        k = self.attn_k(x)  # (B, num_tokens, num_kv_groups * head_dim)
        v = self.attn_v(x)  # (B, num_tokens, num_kv_groups * head_dim)

        # split heads
        q = q.reshape(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)         # (B, n_heads, num_tokens, head_dim)
        k = k.reshape(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)   # (B, num_kv_groups, num_tokens, head_dim)
        v = v.reshape(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)   # (B, num_kv_groups, num_tokens, head_dim)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        position_offset = self.k_cache.shape[2] if self.k_cache is not None else 0
        q = compute_rope(q, cos, sin, position_offset)
        k = compute_rope(k, cos, sin, position_offset)

        if use_cache:
            if self.k_cache is not None and self.v_cache is not None:
                self.k_cache = self.k_cache.cat(k, dim=2).realize()
                self.v_cache = self.v_cache.cat(v, dim=2).realize()
                k = self.k_cache
                v = self.v_cache
            else:
                self.k_cache = k.realize()
                self.v_cache = v.realize()

        # Expand K and V to match number of heads
        # Before:  [[[[0.05, -0.05]],                                  [[0.06, -0.6]]]]
        # After :  [[[[0.05, -0.05]], [[0.05, -0.05]], [[0.06, -0.6]], [[0.06, -0.6]]]]
        k = k.repeat_interleave(self.group_size, dim=1)  # dim=1 <=> num_kv_groups
        v = v.repeat_interleave(self.group_size, dim=1)  # dim=1 <=> num_kv_groups

        # Causal Attention
        if use_cache:
            num_tokens_k, num_tokens_q = k.shape[2], q.shape[2]
            causal_mask = Tensor.triu(Tensor.ones(num_tokens_q, num_tokens_k), diagonal=int(num_tokens_k - num_tokens_q + 1)).bool()
        else:
            causal_mask = Tensor.triu(Tensor.ones(num_tokens, num_tokens), diagonal=1).bool()

        # Attention
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(causal_mask, -math.inf)
        attn_weights = (attn_scores / self.head_dim**0.5).softmax(axis=-1)

        # (B, n_heads, num_tokens, head_dim) -> (B, num_tokens, d_out)
        context_vec = (attn_weights @ v).transpose(1, 2).contiguous().reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock:

    def __init__(self, cfg: dict[str, int | bool | float]):
        super().__init__()

        self.attention = GroupedQueryAttention(
            d_in=int(cfg['emb_dim']),
            n_heads=int(cfg['n_heads']),
            num_kv_groups=int(cfg['n_kv_groups']),
            head_dim=int(cfg['head_dim'])
        )

        self.ff = FeedForward(int(cfg['emb_dim']), int(cfg['hidden_dim']))
        self.norm1 = RMSNorm(int(cfg['emb_dim']))
        self.norm2 = RMSNorm(int(cfg['emb_dim']))

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, use_cache: bool = False) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, cos, sin, use_cache=use_cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x.contiguous()


class TinyQwen:
    def __init__(self, cfg: dict[str, int | bool | float]):
        # Embedding
        self.emb_layer = nn.Embedding(int(cfg['vocab_size']), int(cfg['emb_dim']))

        # Transformer blocks
        self.trf_blocks = [TransformerBlock(cfg) for _ in range(int(cfg["n_layers"]))]

        # Layer norms
        self.final_norm = RMSNorm(int(cfg["emb_dim"]))
        self.out_head = nn.Linear(int(cfg['emb_dim']), int(cfg['vocab_size']), bias=False)

        # ROPE
        self.cos, self.sin = precompute_rope_params(int(cfg['head_dim']), theta_base=cfg['rope_base'], context_length=int(cfg['context_length']))

    def embed(self, in_idx: Tensor) -> Tensor:
        """Token embedding"""
        return self.emb_layer(in_idx).realize()

    def forward(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        x = self.embed(in_idx)
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def __call__(self, in_idx: Tensor, use_cache: bool = False) -> Tensor:
        """Forward"""
        return self.forward(in_idx, use_cache)

    def reset_cache(self) -> None:
        for blk in self.trf_blocks:
            blk.attention.reset_cache()


def assign(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError(f'Dimensions mismatch: {left.shape} != {right.shape}')
    return right.contiguous().realize()


def load_model(model_size: str = '') -> tuple[SimpleTokenizer, TinyQwen]:
    # Load model params
    import pathlib
    ggfu = Tensor(pathlib.Path('Qwen3-0.6B-Q8_0.gguf'))
    kv, state_dict = gguf_load(ggfu.to(None))
    state_dict = {k: v.cast('float16') for k, v in state_dict.items()}
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    model = TinyQwen(QWEN3_CONFIG)

    model.emb_layer.weight = assign(model.emb_layer.weight, state_dict["token_embd.weight"])

    for i, block in enumerate(model.trf_blocks):
        # Layer norms
        block.norm1.scale = assign(block.norm1.scale, state_dict[f"blk.{i}.attn_norm.weight"])
        block.norm2.scale = assign(block.norm2.scale, state_dict[f"blk.{i}.ffn_norm.weight"])

        # Attention
        block.attention.attn_k.weight = assign(block.attention.attn_k.weight, state_dict[f"blk.{i}.attn_k.weight"])
        block.attention.k_norm.scale = assign(block.attention.k_norm.scale, state_dict[f"blk.{i}.attn_k_norm.weight"])

        block.attention.attn_q.weight = assign(block.attention.attn_q.weight, state_dict[f"blk.{i}.attn_q.weight"])
        block.attention.q_norm.scale = assign(block.attention.q_norm.scale, state_dict[f"blk.{i}.attn_q_norm.weight"])

        block.attention.attn_v.weight = assign(block.attention.attn_v.weight, state_dict[f"blk.{i}.attn_v.weight"])

        block.attention.out_proj.weight = assign(block.attention.out_proj.weight, state_dict[f"blk.{i}.attn_output.weight"])

        # FFN
        block.ff.up_proj.weight = assign(block.ff.up_proj.weight, state_dict[f"blk.{i}.ffn_up.weight"])
        block.ff.gate_proj.weight = assign(block.ff.gate_proj.weight, state_dict[f"blk.{i}.ffn_gate.weight"])
        block.ff.down_proj.weight = assign(block.ff.down_proj.weight, state_dict[f"blk.{i}.ffn_down.weight"])

    # Final norm
    model.final_norm.scale = assign(model.final_norm.scale, state_dict["output_norm.weight"])

    # Output head
    model.out_head.weight = assign(model.out_head.weight, state_dict["token_embd.weight"])

    return tokenizer, model
