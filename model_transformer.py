import numpy as np

class TinyTransformerLM:
    """
    Word-level language model with a single self-attention block over the
    context window, followed by a feed-forward layer to predict the next word.

    Architecture:
      token embeddings + positional embeddings
      -> self-attention (single head, Q/K/V + output projection)
      -> flatten attended context
      -> feed-forward (tanh hidden layer)
      -> softmax over vocab
    """

    def __init__(self, vocab_size, embed_dim=330, hidden_dim=330, context_size=10):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        rng = np.random.default_rng(42)
        scale = 0.02

        # token + positional embeddings
        self.embeddings = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32) * scale
        self.pos_embeddings = rng.standard_normal((context_size, embed_dim)).astype(np.float32) * scale

        # single-head self-attention
        self.Wq = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.Wk = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.Wv = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.Wo = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
        self.bq = np.zeros((1, embed_dim), dtype=np.float32)
        self.bk = np.zeros((1, embed_dim), dtype=np.float32)
        self.bv = np.zeros((1, embed_dim), dtype=np.float32)
        self.bo = np.zeros((1, embed_dim), dtype=np.float32)

        # feed-forward head (flattened attended context -> hidden -> vocab)
        self.W1 = rng.standard_normal((embed_dim * context_size, hidden_dim)).astype(np.float32) * scale
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, vocab_size)).astype(np.float32) * scale
        self.b2 = np.zeros((1, vocab_size), dtype=np.float32)

    def num_params(self):
        parts = [self.embeddings, self.pos_embeddings,
                  self.Wq, self.Wk, self.Wv, self.Wo, self.bq, self.bk, self.bv, self.bo,
                  self.W1, self.b1, self.W2, self.b2]
        return sum(p.size for p in parts)

    def softmax(self, x, axis=-1):
        exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp / np.sum(exp, axis=axis, keepdims=True)

    # ---------- batched forward ----------
    def forward_batch(self, token_id_batch):
        """
        token_id_batch: (B, context_size) int array
        Returns cache dict with everything needed for backward, plus probs (B, vocab_size)
        """
        B, T = token_id_batch.shape
        tok_emb = self.embeddings[token_id_batch]                 # (B,T,E)
        x = tok_emb + self.pos_embeddings[None, :T, :]             # (B,T,E)

        Q = x @ self.Wq + self.bq                                  # (B,T,E)
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        scale = 1.0 / np.sqrt(self.embed_dim)
        scores = np.einsum('bte,bse->bts', Q, K) * scale           # (B,T,T)
        attn = self.softmax(scores, axis=-1)                       # (B,T,T)
        attn_out = np.einsum('bts,bse->bte', attn, V)               # (B,T,E)
        attn_out = attn_out @ self.Wo + self.bo                     # (B,T,E)

        # residual connection
        ctx = x + attn_out                                          # (B,T,E)
        flat = ctx.reshape(B, -1)                                   # (B, T*E)

        h = np.tanh(flat @ self.W1 + self.b1)                       # (B,H)
        logits = h @ self.W2 + self.b2                              # (B,V)
        probs = self.softmax(logits, axis=-1)

        cache = dict(token_id_batch=token_id_batch, x=x, Q=Q, K=K, V=V,
                     attn=attn, attn_out=attn_out, ctx=ctx, flat=flat, h=h, probs=probs)
        return cache

    def forward(self, token_ids):
        """Single-example forward for inference."""
        batch = np.array([token_ids], dtype=np.int64)
        cache = self.forward_batch(batch)
        return cache['flat'], cache['h'], cache['probs']

    def train_batch(self, token_id_batch, target_batch, lr=0.01):
        B, T = token_id_batch.shape
        cache = self.forward_batch(token_id_batch)
        probs = cache['probs']

        target_probs = probs[np.arange(B), target_batch]
        loss = -np.mean(np.log(target_probs + 1e-9))

        # ---- backward through FF head ----
        dlogits = probs.copy()
        dlogits[np.arange(B), target_batch] -= 1.0
        dlogits /= B

        dW2 = cache['h'].T @ dlogits
        db2 = dlogits.sum(axis=0, keepdims=True)
        dh = dlogits @ self.W2.T
        dtanh = (1.0 - cache['h'] ** 2) * dh
        dW1 = cache['flat'].T @ dtanh
        db1 = dtanh.sum(axis=0, keepdims=True)
        dflat = dtanh @ self.W1.T                     # (B, T*E)
        dctx = dflat.reshape(B, T, self.embed_dim)     # (B,T,E)

        # ---- backward through residual + attention output proj ----
        dx_resid = dctx.copy()                          # residual path grad
        dattn_out = dctx.copy()                          # into Wo path

        # recompute attn @ V (pre output-projection) for the Wo gradient
        attnV = np.einsum('bts,bse->bte', cache['attn'], cache['V'])
        dWo = attnV.reshape(-1, self.embed_dim).T @ dattn_out.reshape(-1, self.embed_dim)
        dbo = dattn_out.sum(axis=(0, 1), keepdims=False).reshape(1, -1)
        dattnV = dattn_out @ self.Wo.T                   # (B,T,E)

        dattn = np.einsum('bte,bse->bts', dattnV, cache['V'])     # (B,T,T)
        dV = np.einsum('bts,bte->bse', cache['attn'], dattnV)      # (B,T,E)

        # softmax backward (per row over last axis)
        s = cache['attn']
        dscores = s * (dattn - np.sum(dattn * s, axis=-1, keepdims=True))  # (B,T,T)
        scale = 1.0 / np.sqrt(self.embed_dim)
        dscores *= scale

        dQ = np.einsum('bts,bse->bte', dscores, cache['K'])        # (B,T,E)
        dK = np.einsum('bts,bte->bse', dscores, cache['Q'])        # (B,T,E)

        x_flat = cache['x'].reshape(-1, self.embed_dim)
        dQ_flat = dQ.reshape(-1, self.embed_dim)
        dK_flat = dK.reshape(-1, self.embed_dim)
        dV_flat = dV.reshape(-1, self.embed_dim)

        dWq = x_flat.T @ dQ_flat
        dbq = dQ_flat.sum(axis=0, keepdims=True)
        dWk = x_flat.T @ dK_flat
        dbk = dK_flat.sum(axis=0, keepdims=True)
        dWv = x_flat.T @ dV_flat
        dbv = dV_flat.sum(axis=0, keepdims=True)

        dx_from_attn = (dQ_flat @ self.Wq.T + dK_flat @ self.Wk.T + dV_flat @ self.Wv.T).reshape(B, T, self.embed_dim)

        dx = dx_resid + dx_from_attn   # (B,T,E)

        # ---- update params ----
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.Wo -= lr * dWo
        self.bo -= lr * dbo
        self.Wq -= lr * dWq
        self.bq -= lr * dbq
        self.Wk -= lr * dWk
        self.bk -= lr * dbk
        self.Wv -= lr * dWv
        self.bv -= lr * dbv

        # positional embeddings grad
        dpos = dx.sum(axis=0)  # (T,E)
        self.pos_embeddings[:T] -= lr * dpos

        # token embeddings grad (scatter-add)
        np.add.at(self.embeddings, token_id_batch, -lr * dx)

        return loss

    def train_step(self, token_ids, target, lr=0.01):
        return self.train_batch(np.array([token_ids], dtype=np.int64),
                                 np.array([target], dtype=np.int64), lr=lr)

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
