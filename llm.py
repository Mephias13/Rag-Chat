import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # port floats between reboots, see README
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")


def _post(path, body, timeout=600):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def chat(messages, model=CHAT_MODEL, temperature=0.0, num_ctx=4096, keep_alive="5m"):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # qwen3.5 returns empty content otherwise
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "seed": 0, "num_gpu": 99, "num_ctx": num_ctx},
    }
    return _post("/api/chat", body)["message"]["content"].strip()


def embed(texts, keep_alive="5m"):
    body = {"model": EMBED_MODEL, "input": texts, "keep_alive": keep_alive}
    return _post("/api/embed", body)["embeddings"]


def unload(model):
    _post("/api/generate", {"model": model, "keep_alive": 0})


def on_gpu(model):
    """Ollama silently falls back to CPU after restarts; timings from CPU are not comparable."""
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps") as resp:
        loaded = json.load(resp)["models"]
    return any(m["name"] in (model, f"{model}:latest") and m["size_vram"] == m["size"] for m in loaded)
