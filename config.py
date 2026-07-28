# Alysium TinyLM v3 - Character Level
# Change these numbers later to make the model stronger, then retrain.

EMBED_DIM = 64
HIDDEN_DIM = 128
CONTEXT_SIZE = 32          # how many previous characters the model sees
EPOCHS = 15
LEARNING_RATE = 0.03

# Generation
MAX_NEW_CHARS = 80
TEMPERATURE = 0.7
TOP_K = 15
