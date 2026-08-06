"""
Builds a new TinyTransformerLM, warm-starting the token embeddings
from the previously trained MLP model (tinylm.pkl) for any word that
exists in both vocabularies. Everything else (attention weights,
positional embeddings, new vocab rows, FF head) starts randomly
initialized, since there is no equivalent to transfer from the MLP.
"""
import pickle
import re
import numpy as np
from model_transformer import TinyTransformerLM
import config

print("Loading old MLP checkpoint for warm-start...")
with open("tinylm_v1_mlp_backup.pkl", "rb") as f:
    old_ckpt = pickle.load(f)

old_model = old_ckpt["model"]
old_word_to_idx = old_ckpt["word_to_idx"]
old_embeddings = old_model.embeddings  # (old_vocab, old_embed_dim)

print(f"Old model: vocab={len(old_word_to_idx)}, embed_dim={old_model.embed_dim}")

# ---------- Build new vocabulary from the grown dataset ----------
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()
words = re.findall(r"[a-z0-9']+|[.!?]", text)
words = [w for w in words if w.strip()]

vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
PAD = "<pad>"
word_to_idx[PAD] = len(word_to_idx)
idx_to_word[len(idx_to_word)] = PAD
vocab.append(PAD)

print(f"New vocab size: {len(vocab)}")

CONTEXT = config.CONTEXT_SIZE
new_model = TinyTransformerLM(
    vocab_size=len(vocab),
    embed_dim=config.EMBED_DIM,
    hidden_dim=config.HIDDEN_DIM,
    context_size=CONTEXT,
)
print(f"New model params: {new_model.num_params():,}")

# ---------- Warm-start embeddings ----------
# Old embed_dim != new embed_dim in general, so we can only copy the
# overlapping dimensions (min of the two) for words present in both vocabs.
old_E = old_model.embed_dim
new_E = new_model.embed_dim
copy_dim = min(old_E, new_E)

transferred = 0
for w, old_idx in old_word_to_idx.items():
    if w in word_to_idx:
        new_idx = word_to_idx[w]
        new_model.embeddings[new_idx, :copy_dim] = old_embeddings[old_idx, :copy_dim]
        transferred += 1

print(f"Transferred embeddings for {transferred}/{len(word_to_idx)} words "
      f"(copied {copy_dim} of {new_E} embedding dims, rest stays randomly initialized)")

# ---------- Save as the working checkpoint ----------
with open("tinylm.pkl", "wb") as f:
    pickle.dump({
        "model": new_model,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "context_size": CONTEXT,
        "pad_token": PAD,
        "architecture": "transformer",
    }, f)

print("Warm-started checkpoint saved to tinylm.pkl")
