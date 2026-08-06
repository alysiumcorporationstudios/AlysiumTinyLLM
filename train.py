import pickle
import re
import time
import numpy as np
from model_transformer import TinyTransformerLM
import config

print("Alysium TinyLM - Transformer-augmented Training")
print("-" * 50)

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()
words = re.findall(r"[a-z0-9']+|[.!?]", text)
words = [w for w in words if w.strip()]
print(f"Total words: {len(words)}")

# Load warm-started model + vocab (built by warm_start.py)
with open("tinylm.pkl", "rb") as f:
    ckpt = pickle.load(f)

model = ckpt["model"]
word_to_idx = ckpt["word_to_idx"]
idx_to_word = ckpt["idx_to_word"]
CONTEXT = ckpt["context_size"]
PAD = ckpt["pad_token"]
pad_id = word_to_idx[PAD]

print(f"Vocabulary size: {len(word_to_idx)}")
print(f"Model: embed={config.EMBED_DIM}, hidden={config.HIDDEN_DIM}, context={CONTEXT} (transformer)")
print(f"Total parameters: {model.num_params():,}")

BATCH_SIZE = getattr(config, "BATCH_SIZE", 256)

print("\nBuilding training examples...")
word_ids = np.array([word_to_idx.get(w, pad_id) for w in words], dtype=np.int64)
indices = np.arange(0, len(words) - 1, 3)
n = len(indices)
contexts = np.full((n, CONTEXT), pad_id, dtype=np.int64)
targets = np.empty(n, dtype=np.int64)

for row, i in enumerate(indices):
    start = max(0, i - CONTEXT + 1)
    ctx = word_ids[start:i + 1]
    if len(ctx) < CONTEXT:
        contexts[row, CONTEXT - len(ctx):] = ctx
    else:
        contexts[row, :] = ctx[-CONTEXT:]
    targets[row] = word_ids[i + 1]

print(f"Training examples: {n:,}")
print("\nTraining...")

for epoch in range(config.EPOCHS):
    lr = config.LEARNING_RATE * (0.95 ** epoch)
    perm = np.random.permutation(n)
    total_loss = 0.0
    steps = 0
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        batch_idx = perm[start:start + BATCH_SIZE]
        loss = model.train_batch(contexts[batch_idx], targets[batch_idx], lr=lr)
        total_loss += loss
        steps += 1

    avg = total_loss / max(steps, 1)
    elapsed = time.time() - t0
    print(f"Epoch {epoch+1:2d}/{config.EPOCHS}  loss={avg:.4f}  lr={lr:.4f}  time={elapsed:.1f}s")

    with open("tinylm.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "context_size": CONTEXT,
            "pad_token": PAD,
            "architecture": "transformer",
        }, f)

print("\nTraining complete! Saved tinylm.pkl")
