from tinygrad import Tensor, nn
from gelu_vs_silu import SiLU


class FeedForward:

    def __init__(self, emb_dim: int, hidden_dim: int):
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = SiLU()

    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


if __name__ == "__main__":
    emb_dim = 1024
    hidden_dim = 3072
    x = Tensor.randn((5, emb_dim))
    ff = FeedForward(emb_dim, hidden_dim)
    x = ff(x).realize()
    print(x.shape)
