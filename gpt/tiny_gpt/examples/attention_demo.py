from tinygrad import Tensor  # type: ignore
import operator
import math
from itertools import chain
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiny_gpt import TinyEncoder, download_gpt_model, load_model, load_weights  # noqa: E402


def rnd(t: Tensor) -> list[str]:
    length = functools.reduce(operator.mul, t.shape, 1)
    result = ["",] * length

    for i, v in enumerate(chain.from_iterable(t.tolist())):
        result[i] = f"{v:.02f}"
    return result


model = "gpt2"
print(f'Model Size: {model}')
print('Downloading model...')
weights_file = download_gpt_model(model_size=model)

print("Loading weights...")
weights = load_weights(weights_file)

print("Loading model from weights...")
tiny_model = load_model(weights, model)
print('Model loaded.')

encoder = TinyEncoder()
text = "What is the purpose of the universe?"
idx = encoder(text).unsqueeze(0)

print(f"Input : {text}")
print(f"Tokens: {idx.squeeze().tolist()}")

x = tiny_model.embed(idx)
attn = tiny_model.trf_blocks[0].attention

B, num_tokens, d_in = x.shape

# linear projections
qkv = attn.c_attn(x)
queries, keys, values = qkv.chunk(3, dim=-1)

# split heads (B, num_tokens, d_out) -> (B, n_heads, num_tokens, head_dim)
keys = keys.reshape(B, num_tokens, attn.n_heads, attn.head_dim).transpose(1, 2)
queries = queries.reshape(B, num_tokens, attn.n_heads, attn.head_dim).transpose(1, 2)
values = values.reshape(B, num_tokens, attn.n_heads, attn.head_dim).transpose(1, 2)

attention_scores = queries @ keys.transpose(2, 3)

causal_mask = Tensor.triu(Tensor.ones(num_tokens, num_tokens), diagonal=1).bool()

attention_scores = attention_scores.masked_fill(causal_mask, -math.inf)
attention_weights = (attention_scores / attn.head_dim ** 0.5).softmax(axis=-1)
attention_weights = attention_weights.dropout(attn.dropout)  # (1, n_heads, n_tokens, n_tokens)

# Let's see how different heads focus on different tokens for token 0
print("Weights for heads of token 0:")
for i, head in enumerate(attention_weights[:, :, :1].squeeze(0)):
    print(f"Head #{i + 1:02d}: {rnd(head)}")

# Now, let's take a look at the head weights for the first two tokens
print("Weights for heads of token [0:1]:")
for i, head in enumerate(attention_weights[:, :, 1:2].squeeze(0)):
    print(f"Head #{i + 1:02d}: {rnd(head)}")

# Now, let's take a look at the head weights for all tokens
print("Weights for heads of all tokens:")
for i, head in enumerate(attention_weights[:, :, -1:].squeeze(0)):
    print(f"Head #{i + 1:02d}: {rnd(head)}")
