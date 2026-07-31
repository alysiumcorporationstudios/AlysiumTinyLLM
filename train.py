import pickle
import re
import numpy as np
from model import TinyLM
import config

print("Alysium TinyLM - Word Level Training")
print("-" * 50)

# Load text
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Simple word tokenizer
words = re.findall(r"[a-z0-9']+|[.!?]", text)
words = [w for w in words if w.strip()]

print(f"Total words: {len(words)}")

# Build vocabulary
vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

PAD = "<pad>"
word_to_idx[PAD] = len(word_to_idx)
idx_to_word[len(idx_to_word)] = PAD
vocab.append(PAD)

print(f"Vocabulary size: {len(vocab)}")

CONTEXT = config.CONTEXT_SIZE
model = TinyLM(
    vocab_size=len(vocab),
    embed_dim=config.EMBED_DIM,
    hidden_dim=config.HIDDEN_DIM,
    context_size=CONTEXT
)

print(f"Model: embed={config.EMBED_DIM}, hidden={config.HIDDEN_DIM}, context={CONTEXT}")
print("\nTraining...")

for epoch in range(config.EPOCHS):
    total_loss = 0.0
    steps = 0
    lr = config.LEARNING_RATE * (0.95 ** epoch)

    for i in range(0, len(words) - 1, 3):
        start = max(0, i - CONTEXT + 1)
        ctx_words = words[start : i + 1]
        while len(ctx_words) < CONTEXT:
            ctx_words = [PAD] + ctx_words

        context = [word_to_idx[w] for w in ctx_words[-CONTEXT:]]
        target = word_to_idx[words[i + 1]]

        loss = model.train_step(context, target, lr=lr)
        total_loss += loss
        steps += 1

    avg = total_loss / max(steps, 1)
    print(f"Epoch {epoch+1:2d}/{config.EPOCHS}  loss={avg:.4f}  lr={lr:.4f}")

# Save
with open("tinylm.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "context_size": CONTEXT,
        "pad_token": PAD,
    }, f)

print("\nTraining complete! Saved tinylm.pkl")
