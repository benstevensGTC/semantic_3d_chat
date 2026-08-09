"""Oracle-isolated static continuous-scene chat clients."""

import os

# Chat inference is strictly local and never needs an authentication token.
# Set these before importing Transformers/Hugging Face modules so their cache
# resolver neither contacts the network nor consults a user credential file.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
