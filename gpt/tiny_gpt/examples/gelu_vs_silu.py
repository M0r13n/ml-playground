import math
import matplotlib.pyplot as plt  # pip install matplotlib
from tinygrad import Tensor


class GELU:

    def __call__(self, x: Tensor) -> Tensor:
        result = x + 0.044715 * x.pow(3)
        result = (2 / math.pi) ** 0.5 * result
        result = 1 + result.tanh()
        result = 0.5 * x * result
        return result


class SiLU:

    def __call__(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


def linspace(start, end, steps):
    return Tensor.arange(steps) * ((end - start) / (steps - 1)) + start


if __name__ == "__main__":
    # GELU ---
    gelu = GELU()
    x = linspace(-8, 8, 100)
    y_gelu = gelu(x)

    plt.figure(figsize=(4, 3))
    plt.title("TinyGELU activation function")
    plt.xlabel("x")
    plt.ylabel("GELU(x)")
    plt.grid(True)
    plt.plot(x.numpy(), y_gelu.numpy())
    plt.tight_layout()
    plt.savefig('gelu.png')

    # SiLU ---
    silu = SiLU()
    y_silu = silu(x)

    plt.figure(figsize=(4, 3))
    plt.title("SiLU activation function")
    plt.xlabel("x")
    plt.ylabel("SiLU(x)")
    plt.grid(True)
    plt.plot(x.numpy(), y_silu.numpy())
    plt.tight_layout()
    plt.savefig('silu.png')

    print(y_gelu[:10].tolist())
    print(y_silu[:10].tolist())
