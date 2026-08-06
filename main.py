import re
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import TinyLM
import config

# ---------- Load model once at startup ----------
print("Loading Alysium TinyLM (word-level)...")

with open("tinylm.pkl", "rb") as f:
    ckpt = pickle.load(f)

model = ckpt["model"]
word_to_idx = ckpt["word_to_idx"]
idx_to_word = ckpt["idx_to_word"]
ctx_size = ckpt["context_size"]
PAD = ckpt["pad_token"]
pad_id = word_to_idx[PAD]

print(f"Model loaded. params={model.num_params():,} vocab={len(word_to_idx)} context={ctx_size}")

# ---------- FastAPI app ----------
app = FastAPI(
    title="Alysium TinyLM API",
    description="Lightweight word-level language model",
    version="4.0"
)

# Allow your frontend / Neuraprompt AI to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    max_new_words: int = config.MAX_NEW_WORDS
    temperature: float = config.TEMPERATURE
    top_k: int = config.TOP_K


class ChatResponse(BaseModel):
    reply: str


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9']+|[.!?]", text.lower()) if w.strip()]


def generate(prompt: str, max_new: int = 12, temperature: float = 0.45, top_k: int = 8) -> str:
    words = tokenize(prompt)
    ids = [word_to_idx.get(w, pad_id) for w in words]

    if len(ids) < ctx_size:
        ids = [pad_id] * (ctx_size - len(ids)) + ids
    else:
        ids = ids[-ctx_size:]

    generated = []
    for _ in range(max_new):
        _, _, probs = model.forward(ids)
        next_id = model.sample(probs, temperature=temperature, top_k=top_k)
        next_word = idx_to_word.get(next_id, "")
        if next_word and next_word != PAD:
            generated.append(next_word)
        ids = ids[1:] + [next_id]

    # join words, tidy spacing around punctuation
    text = " ".join(generated)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    return text.strip()


@app.get("/")
def home():
    return {
        "name": "Alysium TinyLM",
        "version": "4.0",
        "type": "word-level",
        "vocab_size": len(word_to_idx),
        "parameters": model.num_params(),
        "status": "running",
        "docs": "/docs"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        reply = generate(
            req.message,
            max_new=req.max_new_words,
            temperature=req.temperature,
            top_k=req.top_k
        )
        if not reply:
            reply = "..."
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
