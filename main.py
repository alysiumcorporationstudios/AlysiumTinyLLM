"""
Alysium TinyLM — hosted inference server (v5) for DeepImagix AI + Neuraprompt AI.

Deployed on Render (https://alysiumtinyllm.onrender.com).

Contract (matches com.alysium.corp.studios.deepimagix.ai.data.api.TinyLmApi):

  POST /chat
    Request:  {"user_id": "...", "message": "...", "consent": true}
    Response: {"response": "...", "trained_on": true, "flavor": "tinylm"}

  - If consent is false (or the user is over the daily quota), the message is
    NOT saved to the training set and trained_on comes back false.
  - Consented messages are appended to training_log.jsonl.
  - "reply" is also returned for backward compatibility with Neuraprompt AI.

  GET  /              -> service info
  GET  /health        -> {"status": "ok"}
  GET  /training-log  -> consented messages collected so far (JSONL)
  GET  /docs          -> auto-generated OpenAPI docs
"""
import json
import os
import re
import time
import pickle
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from model_transformer import TinyTransformerLM
import config

# ---------- Load model once at startup ----------
print("Loading Alysium TinyLM checkpoint...", flush=True)

CKPT_PATH = os.environ.get("TINYLM_CKPT", "tinylm.pkl")
with open(CKPT_PATH, "rb") as f:
    ckpt = pickle.load(f)

model = ckpt["model"]
word_to_idx = ckpt["word_to_idx"]
idx_to_word = ckpt["idx_to_word"]
ctx_size = ckpt["context_size"]
PAD = ckpt["pad_token"]
pad_id = word_to_idx[PAD]

print(
    f"Model loaded: {type(model).__name__}, params={model.num_params():,}, "
    f"vocab={len(word_to_idx)}, context={ctx_size}",
    flush=True,
)

# ---------- Training-consent log ----------
LOG_PATH = Path(os.environ.get("TINYLM_LOG", "training_log.jsonl"))

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
DATASET_REPO = os.environ.get("TINYLM_DATASET_REPO", "").strip()

DAILY_USER_QUOTA = int(os.environ.get("TINYLM_DAILY_QUOTA", "300"))
_day_counts: dict = defaultdict(int)
_day = time.strftime("%Y-%m-%d")


def _today() -> str:
    global _day
    now = time.strftime("%Y-%m-%d")
    if now != _day:
        _day = now
        _day_counts.clear()
    return _day


def _persist(example: dict) -> bool:
    """Append consented example to the JSONL training log."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not write training log: {e}", flush=True)
        return False
    if HF_TOKEN and DATASET_REPO:
        try:
            from huggingface_hub import HfApi
            HfApi(token=HF_TOKEN).upload_file(
                path_or_fileobj=str(LOG_PATH),
                path_in_repo="training_log.jsonl",
                repo_id=DATASET_REPO,
                repo_type="dataset",
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not mirror log to HF dataset: {e}", flush=True)
    return True


# ---------- FastAPI app ----------
app = FastAPI(
    title="Alysium TinyLM API",
    description="Lightweight pure-NumPy word-level transformer LM, hosted for the DeepImagix AI app.",
    version="5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str = Field(default="", description="Stable per-user id sent by the app")
    message: str = Field(min_length=1, description="The user's chat message")
    consent: bool = Field(default=False, description="User consented to saving this message for training")
    # Optional inference tuning (defaults mirror config.py)
    max_new_words: int = Field(default=config.MAX_NEW_WORDS, ge=1, le=64)
    temperature: float = Field(default=config.TEMPERATURE, gt=0.0, le=2.0)
    top_k: int = Field(default=config.TOP_K, ge=1, le=64)


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+|[.!?]", text.lower())


def generate(prompt: str, max_new: int, temperature: float, top_k: int) -> str:
    words = tokenize(prompt)
    ids = [word_to_idx.get(w, pad_id) for w in words]
    ids = ids[-ctx_size:]
    ids = [pad_id] * (ctx_size - len(ids)) + ids

    generated: list = []
    for _ in range(max_new):
        _, _, probs = model.forward(ids)
        next_id = model.sample(probs, temperature=temperature, top_k=top_k)
        next_word = idx_to_word.get(next_id, "")
        if not next_word or next_word == PAD:
            break
        generated.append(next_word)
        if next_word in (".", "!", "?"):
            break
        ids = ids[1:] + [next_id]

    text = re.sub(r"\s+([.!?])", r"\1", " ".join(generated))
    return text.strip()


@app.get("/")
def home():
    return {
        "name": "Alysium TinyLM",
        "version": "5.0",
        "architecture": type(model).__name__,
        "parameters": model.num_params(),
        "vocab_size": len(word_to_idx),
        "context_window": ctx_size,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/training-log", response_class=PlainTextResponse)
def training_log():
    if LOG_PATH.exists():
        return PlainTextResponse(LOG_PATH.read_text(encoding="utf-8"), media_type="text/plain")
    return PlainTextResponse("", media_type="text/plain")


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    reply = generate(message, req.max_new_words, req.temperature, req.top_k)
    if not reply:
        reply = "hmm i am not sure."

    # --- training consent handling (consent must be explicit) ---
    trained_on = False
    if req.consent:
        _today()
        if _day_counts[req.user_id or "anon"] >= DAILY_USER_QUOTA:
            trained_on = False  # quota reached — answer anyway, don't log
        else:
            example = {
                "user_id": req.user_id,
                "prompt": message,
                "response": reply,
                "timestamp": int(time.time()),
            }
            if _persist(example):
                trained_on = True
                _day_counts[req.user_id or "anon"] += 1

    return {
        "response": reply,
        "reply": reply,  # backward-compatible with Neuraprompt AI (main.py v4)
        "trained_on": trained_on,
        "flavor": "tinylm",
    }
