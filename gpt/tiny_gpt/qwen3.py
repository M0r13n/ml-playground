import math
from tinygrad import Tensor, nn
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


def load_model():
    # Load model params
    ggfu = Tensor.from_url("https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf")
    return gguf_load(ggfu.to(None))


def main():

    kv, state_dict = load_model()
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)

    some_tokens = tokenizer.encode("Hello, World!")
    print(state_dict)
    print(some_tokens)
    print(tokenizer.decode(some_tokens))


if __name__ == "__main__":
    main()
