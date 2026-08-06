"""
Chunked trainer: runs for a wall-clock time budget, then saves checkpoint
and exits, recording progress (epoch, position within epoch) so it can be
resumed by running this script again.
"""
import pickle
import re
import sys
import time
import numpy as np
from model_transformer import TinyTransformerLM
import config

TIME_BUDGET_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()
words = re.findall(r"[a-z0-9']+|[.!?]", text)
words = [w for w in words if w.strip()]

with open("tinylm.pkl", "rb") as f:
    ckpt = pickle.load(f)

model = ckpt["model"]
word_to_idx = ckpt["word_to_idx"]
idx_to_word = ckpt["idx_to_word"]
CONTEXT = ckpt["context_size"]
PAD = ckpt["pad_token"]
pad_id = word_to_idx[PAD]

progress = ckpt.get("progress", {"epoch": 0, "batch_start": 0, "loss_history": []})

BATCH_SIZE = getattr(config, "BATCH_SIZE", 256)

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

print(f"Total examples per epoch: {n:,}, batch size: {BATCH_SIZE}")
print(f"Resuming at epoch {progress['epoch']+1}/{config.EPOCHS}, batch_start={progress['batch_start']}")

# stable shuffle per epoch, seeded by epoch number so resuming an epoch uses the SAME order
epoch = progress["epoch"]
if epoch >= config.EPOCHS:
    print("Training already complete.")
    sys.exit(0)

rng = np.random.default_rng(1000 + epoch)
perm = rng.permutation(n)

lr = config.LEARNING_RATE * (0.95 ** epoch)
batch_start = progress["batch_start"]
total_loss = progress.get("epoch_loss_sum", 0.0)
steps = progress.get("epoch_steps", 0)

t0 = time.time()
pos = batch_start
while pos < n:
    if time.time() - t0 > TIME_BUDGET_SEC:
        break
    batch_idx = perm[pos:pos + BATCH_SIZE]
    loss = model.train_batch(contexts[batch_idx], targets[batch_idx], lr=lr)
    total_loss += loss
    steps += 1
    pos += BATCH_SIZE

elapsed = time.time() - t0
pct = 100.0 * pos / n
print(f"Epoch {epoch+1}/{config.EPOCHS}  processed {pos:,}/{n:,} ({pct:.1f}%)  "
      f"running_avg_loss={total_loss/max(steps,1):.4f}  chunk_time={elapsed:.1f}s")

if pos >= n:
    # epoch finished
    print(f"==> Epoch {epoch+1} COMPLETE. avg_loss={total_loss/max(steps,1):.4f}")
    progress["loss_history"].append(total_loss / max(steps, 1))
    progress["epoch"] = epoch + 1
    progress["batch_start"] = 0
    progress["epoch_loss_sum"] = 0.0
    progress["epoch_steps"] = 0
else:
    progress["batch_start"] = pos
    progress["epoch_loss_sum"] = total_loss
    progress["epoch_steps"] = steps

with open("tinylm.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "context_size": CONTEXT,
        "pad_token": PAD,
        "architecture": "transformer",
        "progress": progress,
    }, f)

if progress["epoch"] >= config.EPOCHS:
    print("\nALL EPOCHS COMPLETE.")
else:
    print("Checkpoint saved. Run again to continue.")
