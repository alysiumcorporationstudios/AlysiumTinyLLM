import pickle
import re
import numpy as np
from model import TinyLM
import config

with open("tinylm.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
word_to_idx = data["word_to_idx"]
idx_to_word = data["idx_to_word"]
CONTEXT = data["context_size"]
PAD = data["pad_token"]
PAD_ID = word_to_idx[PAD]

print("Alysium TinyLM (Word Level)")
print("Type a message (or 'quit' to exit)")
print("-" * 40)


def tokenize(text):
    return re.findall(r"[a-z0-9']+|[.!?]", text.lower().strip())


def words_to_context(word_list):
    ids = [word_to_idx[w] for w in word_list if w in word_to_idx]
    ids = ids[-CONTEXT:]
    while len(ids) < CONTEXT:
        ids = [PAD_ID] + ids
    return ids


def generate(prompt_words, max_new=15, temperature=0.5, top_k=8):
    context_ids = words_to_context(prompt_words)
    generated = []

    for _ in range(max_new):
        _, _, probs = model.forward(context_ids)
        next_id = model.sample(probs, temperature=temperature, top_k=top_k)
        next_word = idx_to_word[next_id]

        if next_word == PAD:
            break

        generated.append(next_word)

        # Stop cleanly at end of sentence
        if next_word in [".", "!", "?"]:
            break

        context_ids = context_ids[1:] + [next_id]

    return generated


while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user:
        continue
    if user.lower() in ("quit", "exit", "bye"):
        print("AI: goodbye.")
        break

    prompt_words = tokenize(user)
    reply_words = generate(prompt_words)

    if not reply_words:
        print("AI: hmm i am not sure.")
    else:
        reply = " ".join(reply_words)
        reply = re.sub(r"\s+([.!?])", r"\1", reply)
        print("AI:", reply)
