import numpy as np

class TinyLM:
    """
    Word-level language model.
    Predicts the next word from the previous CONTEXT_SIZE words.
    Supports batched forward/backward passes for fast training.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, context_size=5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        rng = np.random.default_rng(42)
        self.embeddings = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32) * 0.02
        self.W1 = rng.standard_normal((embed_dim * context_size, hidden_dim)).astype(np.float32) * 0.02
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, vocab_size)).astype(np.float32) * 0.02
        self.b2 = np.zeros((1, vocab_size), dtype=np.float32)

    def num_params(self):
        return (self.embeddings.size + self.W1.size + self.b1.size
                + self.W2.size + self.b2.size)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def forward(self, token_ids):
        """Single-example forward pass (used at inference time)."""
        embs = [self.embeddings[t] for t in token_ids]
        x = np.concatenate(embs).reshape(1, -1)
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        probs = self.softmax(logits)
        return x, h, probs

    def forward_batch(self, token_id_batch):
        """
        token_id_batch: (B, context_size) int array
        Returns x (B, embed_dim*context_size), h (B, hidden_dim), probs (B, vocab_size)
        """
        B = token_id_batch.shape[0]
        x = self.embeddings[token_id_batch].reshape(B, -1)
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits)
        return x, h, probs

    def train_step(self, token_ids, target, lr=0.01):
        """Single-example training step (kept for compatibility)."""
        return self.train_batch(
            np.array([token_ids], dtype=np.int64),
            np.array([target], dtype=np.int64),
            lr=lr,
        )

    def train_batch(self, token_id_batch, target_batch, lr=0.01):
        """
        Vectorized batched training step.
        token_id_batch: (B, context_size) int array
        target_batch: (B,) int array
        Returns mean loss over the batch.
        """
        B = token_id_batch.shape[0]
        x, h, probs = self.forward_batch(token_id_batch)

        target_probs = probs[np.arange(B), target_batch]
        loss = -np.mean(np.log(target_probs + 1e-9))

        dout = probs.copy()
        dout[np.arange(B), target_batch] -= 1.0
        dout /= B

        dW2 = h.T @ dout
        db2 = dout.sum(axis=0, keepdims=True)
        dh = dout @ self.W2.T
        dtanh = (1.0 - h * h) * dh
        dW1 = x.T @ dtanh
        db1 = dtanh.sum(axis=0, keepdims=True)
        dx = dtanh @ self.W1.T

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        dx = dx.reshape(B, self.context_size, self.embed_dim)
        np.add.at(self.embeddings, token_id_batch, -lr * dx)

        return loss

    def sample(self, probs, temperature=0.6, top_k=10):
        logits = np.log(np.clip(probs[0], 1e-9, 1.0)) / max(temperature, 0.05)
        if top_k and top_k < len(logits):
            idx = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e9)
            mask[idx] = logits[idx]
            logits = mask
        exp = np.exp(logits - np.max(logits))
        p = exp / np.sum(exp)
        return int(np.random.choice(len(p), p=p))
