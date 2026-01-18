#!/usr/bin/env python3

import http.server
import pathlib
import json
import sys
import typing

from tinygrad import Device  # type: ignore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from tiny_gpt import TinyEncoder, TinyGPTModel, download_gpt_model, load_model, load_weights, prevent_ngram_repetition  # noqa: E402

MAX_NEW_TOKENS = 200


class ChatServer(http.server.HTTPServer):
    """HTTP server with model injection."""

    def __init__(self, address, handler_class, model):
        super().__init__(address, handler_class)
        self.model = model
        self.encoder = TinyEncoder()


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file("index.html", "text/html")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/chat":
            self.handle_chat()
        else:
            self.send_error(404)

    def serve_file(self, path, content_type):
        safe_path = pathlib.Path(path).resolve()
        base_dir = pathlib.Path(__file__).parent.resolve()
        if not str(safe_path).startswith(str(base_dir)):
            self.send_error(403)
            return
        try:
            with open(safe_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def read_messages(self):
        try:
            # Read request body
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            messages = data.get("messages", [])
            return messages
        except (json.JSONDecodeError, ValueError) as e:
            self.log_error(str(e))
            return None

    def handle_chat(self):
        messages = self.read_messages()
        if not messages:
            self.send_error(400, "Invalid message format.")
            return None
        try:
            last_message = messages[-1]['content']
        except (IndexError, KeyError):
            self.send_error(400, "Invalid message format.")
            return None

        # Prepare response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            # Run inference and stream the output token by token
            self.server.model.reset_cache()
            output = run_inference(self.server.model, last_message, max_new_tokens=MAX_NEW_TOKENS, encoder=self.server.encoder, temperature=0.0)

            visited_chars = 0
            for text in output:
                text, visited_chars = text[visited_chars:], len(text)
                if text == "<|endoftext|>":
                    break
                chunk = f"data: {json.dumps({'text': text})}\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            self.log_error("Client closed connection unexpectedly.")
        self.close_connection = True


def run_inference(model: TinyGPTModel, text: str, max_new_tokens: int, encoder: TinyEncoder, temperature: float = 0.0) -> typing.Generator[str, None, None]:
    """Run inference."""
    idx = encoder(text).unsqueeze(0)
    logits = model(idx, use_cache=True)
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]

        logits = prevent_ngram_repetition(logits, idx, n=5)

        if temperature < 1e-6:
            nxt = logits.argmax(axis=-1, keepdim=True)
        else:
            probas = (logits / temperature).softmax(axis=-1)
            nxt = probas.multinomial()

        nxt = nxt.realize()
        idx = idx.cat(nxt, dim=1)
        logits = model(nxt, use_cache=True).realize()
        output = encoder.tokenizer.decode(idx.squeeze(0).tolist())
        yield output


if __name__ == "__main__":
    print(f"COGENT Terminal — Server")

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

    port = 8080
    server = ChatServer(("", port), Handler, tiny_model)
    print(f"Serving DIALOG/1 at: http://localhost:{port}")
    print("-" * 80)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
