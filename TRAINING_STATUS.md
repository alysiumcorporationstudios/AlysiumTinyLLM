# Training Status — COMPLETE

## Final model
- Architecture: Transformer (single-head self-attention), see `model_transformer.py`
- Parameters: 2,003,487
- Vocabulary: 717 words (word-level tokenization)
- Context window: 10 words
- Dataset: dataset.txt, ~2,000,000 words

## Training result — all 6/6 epochs complete
| Epoch | Avg Loss |
|-------|----------|
| 1     | 3.4279   |
| 2     | 2.2139   |
| 3     | 1.8580   |
| 4     | 1.7196   |
| 5     | 1.6560   |
| 6     | 1.6044   |

Steady, stable convergence throughout — no divergence or instability.

## Files
- `model_transformer.py` — the trained architecture (transformer, active).
- `model.py` — older MLP architecture, kept for reference only; NOT what
  tinylm.pkl currently holds.
- `tinylm.pkl` — final trained checkpoint. Contains: model, word_to_idx,
  idx_to_word, context_size, pad_token, architecture tag, progress log.
- `tinylm_v1_mlp_backup.pkl` — earlier fully-trained MLP model (1.47M
  params, final loss 1.6863) from before the transformer upgrade, kept
  as a fallback.
- `warm_start.py` — script that transferred trained token embeddings
  from the old MLP into this transformer before training began.
- `main.py` — FastAPI server, loads tinylm.pkl directly (word-level).
- `train_chunked.py` — resumable, time-boxed trainer. Not needed for
  further use unless you retrain — training is finished.

## Resuming further training (optional)
Not required — training is complete. If you want to train more epochs,
edit EPOCHS in config.py to a higher number and run:

    python3 train_chunked.py 165

It will pick up from progress stored in tinylm.pkl and continue.
