import textwrap
from tinygrad import Device  # type: ignore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiny_gpt import TinyEncoder, TinyGPTModel, download_gpt_model, load_model, load_weights, prevent_ngram_repetition  # noqa: E402


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


def run_inference(model: TinyGPTModel, text: str, max_new_tokens: int, encoder: TinyEncoder, temperature: float = 0.0) -> None:
    """Run inference."""
    idx = encoder(text).unsqueeze(0)
    logits = model(idx, use_cache=True)
    last_line_count = 0
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]

        logits = prevent_ngram_repetition(logits, idx, n=5)

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
        output = encoder.tokenizer.decode(idx.squeeze(0).tolist())
        text = wrap_text(output)
        last_line_count = print_inplace(text, last_line_count)


def main() -> None:
    model = "gpt2"  # gpt2, gpt2-medium, gpt2-large
    print(f'Default Device: {Device.DEFAULT}')
    print(f'Model Size: {model}')
    print('Downloading model...')
    weights_file = download_gpt_model(model_size=model)

    print("Loading weights...")
    weights = load_weights(weights_file)

    print("Loading model from weights...")
    tiny_model = load_model(weights, model)
    tiny_model.reset_cache()
    print('Model loaded.')

    print("Inference...")
    encoder = TinyEncoder()
    text = sys.argv[1] if len(sys.argv) > 1 else "What is the purpose of life?"

    run_inference(tiny_model, text, max_new_tokens=50, encoder=encoder, temperature=0.0)
    tiny_model.reset_cache()


if __name__ == "__main__":
    main()
