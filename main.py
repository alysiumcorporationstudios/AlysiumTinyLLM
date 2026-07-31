from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

from model import TinyLM
from tokenizer import CharTokenizer
import config

# ---------- Load model once at startup ----------
print("Loading Alysium TinyLM...")
tokenizer = CharTokenizer().load("tokenizer.json")

with open("tinylm.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
ctx_size = data["config"]["context_size"]
print("Model loaded successfully.")

# ---------- FastAPI app ----------
app = FastAPI(
    title="Alysium TinyLM API",
    description="Lightweight character-level language model",
    version="3.0"
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
    max_new_chars: int = 80
    temperature: float = 0.7
    top_k: int = 15

class ChatResponse(BaseModel):
    reply: str


def generate(prompt: str, max_new: int = 80, temperature: float = 0.4, top_k: int = 15) -> str:
    ids = tokenizer.encode(prompt.lower())

    # pad or truncate to context size
    if len(ids) < ctx_size:
        pad_id = tokenizer.char_to_id.get(" ", 0)
        ids = [pad_id] * (ctx_size - len(ids)) + ids
    else:
        ids = ids[-ctx_size:]

    generated = []
    for _ in range(max_new):
        _, _, probs = model.forward(ids)
        next_id = model.sample(probs, temperature=temperature, top_k=top_k)
        next_char = tokenizer.id_to_char.get(next_id, "?")
        generated.append(next_char)
        ids = ids[1:] + [next_id]

    return "".join(generated).strip()


@app.get("/")
def home():
    return {
        "name": "Alysium TinyLM",
        "version": "3.0",
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
            max_new=req.max_new_chars,
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
