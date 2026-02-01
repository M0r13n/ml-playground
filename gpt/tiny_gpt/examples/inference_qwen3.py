import pathlib
import sys
import textwrap
from tinygrad import Device, Tensor
from tinygrad.apps.llm import SimpleTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qwen3 import load_model, TinyQwen  # noqa: E402


def print_inplace(text: str, prev_lines: int) -> int:
    # Clear previous block
    if prev_lines > 0:
        sys.stdout.write(f"\033[{prev_lines}A")
        sys.stdout.write("\033[J")

    print(text, end="\r")
    sys.stdout.flush()
    return text.count("\n")


def wrap_text(text: str, width: int = 80) -> str:
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(lines)


def run_inference(model: TinyQwen, text: str, max_new_tokens: int, tokenizer: SimpleTokenizer, temperature: float = 0.0) -> None:
    """Run inference."""
    idx = Tensor(tokenizer.encode(text)).unsqueeze(0)
    logits = model(idx, use_cache=True)
    last_line_count = 0
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]

        # logits = prevent_ngram_repetition(logits, idx, n=5)

        if temperature < 1e-6:
            nxt = logits.argmax(axis=-1, keepdim=True)
        else:
            # Softmax exponentiation is non-linear.
            # High temperature -> flatter distribution (uniform-like)
            probas = (logits / temperature).softmax(axis=-1)
            # Pick a token based on the probability distribution
            # e.g. random.choices(range(0,len(logits)), weights=logits, k=100)
            nxt = probas.multinomial()

        nxt = nxt.realize()
        idx = idx.cat(nxt, dim=1)
        logits = model(nxt, use_cache=True).realize()
        output = tokenizer.decode(idx.squeeze(0).tolist())
        text = wrap_text(output)
        last_line_count = print_inplace(text, last_line_count)


def main() -> None:
    print(f'Default Device: {Device.DEFAULT}')
    print('Loading model...')
    tokenizer, model = load_model()
    model.reset_cache()
    print('Model loaded.')

    print("Inference...")
    text = sys.argv[1] if len(sys.argv) > 1 else "What is the purpose of life?"

    run_inference(model, text, max_new_tokens=50, tokenizer=tokenizer, temperature=0.0)


if __name__ == "__main__":
    main()
