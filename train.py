import pickle
import re
import time
import numpy as np
from model import TinyLM
import config

print("Alysium TinyLM - Word Level Training (vectorized/batched)")
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
BATCH_SIZE = getattr(config, "BATCH_SIZE", 256)

model = TinyLM(
    vocab_size=len(vocab),
    embed_dim=config.EMBED_DIM,
    hidden_dim=config.HIDDEN_DIM,
    context_size=CONTEXT
)

print(f"Model: embed={config.EMBED_DIM}, hidden={config.HIDDEN_DIM}, context={CONTEXT}")
print(f"Total parameters: {model.num_params():,}")
print(f"Batch size: {BATCH_SIZE}")

# ---------- Pre-build all (context, target) pairs as arrays ----------
print("\nBuilding training examples...")
pad_id = word_to_idx[PAD]
word_ids = np.array([word_to_idx[w] for w in words], dtype=np.int64)

# stride of 3, matching original sampling density
indices = np.arange(0, len(words) - 1, 3)

n = len(indices)
contexts = np.full((n, CONTEXT), pad_id, dtype=np.int64)
targets = np.empty(n, dtype=np.int64)

for row, i in enumerate(indices):
    start = max(0, i - CONTEXT + 1)
    ctx = word_ids[start:i + 1]
    if len(ctx) < CONTEXT:
        pad_amt = CONTEXT - len(ctx)
        contexts[row, pad_amt:] = ctx
    else:
        contexts[row, :] = ctx[-CONTEXT:]
    targets[row] = word_ids[i + 1]

print(f"Training examples: {n:,}")
print("\nTraining...")

for epoch in range(config.EPOCHS):
    lr = config.LEARNING_RATE * (0.95 ** epoch)

    # shuffle each epoch
    perm = np.random.permutation(n)
    total_loss = 0.0
    steps = 0
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        batch_idx = perm[start:start + BATCH_SIZE]
        ctx_batch = contexts[batch_idx]
        tgt_batch = targets[batch_idx]

        loss = model.train_batch(ctx_batch, tgt_batch, lr=lr)
        total_loss += loss
        steps += 1

    avg = total_loss / max(steps, 1)
    elapsed = time.time() - t0
    print(f"Epoch {epoch+1:2d}/{config.EPOCHS}  loss={avg:.4f}  lr={lr:.4f}  time={elapsed:.1f}s")

    # checkpoint after every epoch in case of interruption
    with open("tinylm.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "context_size": CONTEXT,
            "pad_token": PAD,
        }, f)

print("\nTraining complete! Saved tinylm.pkl")
