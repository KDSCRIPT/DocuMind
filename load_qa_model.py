"""
load_qa_model.py

This module loads a fine-tuned text-to-text question answering model using Hugging Face Transformers.
It returns the tokenizer and a pipeline object for generating answers based on input prompts.

Purpose:
Encapsulates the loading of a local QA model and prepares it for inference.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from constants import TEXT_TO_TEXT_MODEL_PATH

def load_qa_model():
    """
    Loads the fine-tuned QA model and tokenizer from the either local directory or from HuggingFace.

    Explanation:
    - Loads the tokenizer and model from the local directory or from HuggingFace..
    - Wraps the model in a Hugging Face `pipeline` for text-to-text generation.
    - Returns both the tokenizer (for encoding/decoding) and the QA pipeline (for inference).

    Returns:
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for input processing.
    - qa_model (transformers.Pipeline): The QA model wrapped in a text2text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(TEXT_TO_TEXT_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_TO_TEXT_MODEL_PATH)
    qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return tokenizer, qa_model
