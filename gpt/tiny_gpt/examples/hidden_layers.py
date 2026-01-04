from tinygrad import Device  # type: ignore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiny_gpt import TinyEncoder, download_gpt_model, explore_hidden_layers, load_model, load_weights  # noqa: E402


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
    text = "What is the purpose of life?"

    print('Exploring hidden layers..')
    explore_hidden_layers(tiny_model, text, max_new_tokens=10, encoder=encoder)


if __name__ == "__main__":
    main()
