"""
download_models.py

This script downloads and saves pretrained models locally for offline usage of the chatbot system.

It fetches:
1. A Sentence Transformer model for embedding text chunks and queries.
2. A Text-to-Text model (like T5) used for generating answers.

By default, it uses HuggingFace model identifiers defined in constants.py, but users must update
`SENTENCE_TRANSFOMER_PATH` and `TEXT_TO_TEXT_MODEL_PATH` to point to valid HuggingFace models before running.

After execution, the models are saved locally under the `LOCAL_MODELS_DIRECTORY` path so the chatbot can function without internet access.

Note:
- You must have internet access the first time you run this script.
- Update the constants if you use different model names or paths.

"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from constants import (
    LOCAL_MODELS_DIRECTORY,
    LOCAL_SENTENCE_TRANSFOMER_PATH,
    LOCAL_TEXT_TO_TEXT_MODEL_PATH,
    SENTENCE_TRANSFOMER_PATH,
    TEXT_TO_TEXT_MODEL_PATH
)
import os

# Create local directories for both models
os.makedirs(f"{LOCAL_MODELS_DIRECTORY}/{LOCAL_SENTENCE_TRANSFOMER_PATH}", exist_ok=True)
os.makedirs(f"{LOCAL_MODELS_DIRECTORY}/{LOCAL_TEXT_TO_TEXT_MODEL_PATH}", exist_ok=True)

# Download and save the Sentence Transformer model for local use
embedder = SentenceTransformer(SENTENCE_TRANSFOMER_PATH)
embedder.save(f"{LOCAL_MODELS_DIRECTORY}/{LOCAL_SENTENCE_TRANSFOMER_PATH}")

# Download and save the QA (text2text) model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEXT_TO_TEXT_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_TO_TEXT_MODEL_PATH)

tokenizer.save_pretrained(f"{LOCAL_MODELS_DIRECTORY}/{LOCAL_TEXT_TO_TEXT_MODEL_PATH}")
model.save_pretrained(f"{LOCAL_MODELS_DIRECTORY}/{LOCAL_TEXT_TO_TEXT_MODEL_PATH}")