import numpy as np

class TinyLM:
    """
    Word-level language model.
    Predicts the next word from the previous CONTEXT_SIZE words.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, context_size=5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.02
        self.W1 = np.random.randn(embed_dim * context_size, hidden_dim) * 0.02
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.02
        self.b2 = np.zeros((1, vocab_size))

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def forward(self, token_ids):
        embs = [self.embeddings[t] for t in token_ids]
        x = np.concatenate(embs).reshape(1, -1)
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        probs = self.softmax(logits)
        return x, h, probs

    def train_step(self, token_ids, target, lr=0.01):
        x, h, probs = self.forward(token_ids)
        loss = -np.log(probs[0, target] + 1e-9)

        dout = probs.copy()
        dout[0, target] -= 1.0

        dW2 = np.dot(h.T, dout)
        db2 = dout
        dh = np.dot(dout, self.W2.T)
        dtanh = (1.0 - h * h) * dh
        dW1 = np.dot(x.T, dtanh)
        db1 = dtanh
        dx = np.dot(dtanh, self.W1.T).reshape(self.context_size, self.embed_dim)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        for i, t in enumerate(token_ids):
            self.embeddings[t] -= lr * dx[i]

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
