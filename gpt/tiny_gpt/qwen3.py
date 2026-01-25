from tinygrad import Tensor
from tinygrad.nn.state import gguf_load
from tinygrad.apps.llm import SimpleTokenizer


class RMSNorm:
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        self.scale = Tensor.ones(emb_dim, requires_grad=True)  # scale but no shift
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(axis=-1)
        return x * (variance + self.eps).rsqrt() * self.scale


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
