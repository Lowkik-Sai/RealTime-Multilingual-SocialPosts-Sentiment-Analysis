import re
import warnings
from transformers import pipeline

# Global cache for the translation pipeline
_translator_pipeline = None
_translation_cache = {}

def get_translator():
    """Lazily loads and caches the translation model."""
    global _translator_pipeline
    if _translator_pipeline is None:
        # This model is excellent for translating many languages to English.
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        _translator_pipeline = pipeline("translation", model=model_name, device=-1)
    return _translator_pipeline

def translate_text(text: str) -> str:
    """Translates a text to English using a cached pipeline."""
    if not text or not isinstance(text, str):
        return ""
    try:
        if text in _translation_cache:
            return _translation_cache[text]
        translator = get_translator()
        # The pipeline returns a list with a dictionary
        result = translator(text, max_length=512)
        _translation_cache[text] = result[0]['translation_text']
        return result[0]['translation_text']

    except Exception as e:
        warnings.warn(f"Translation failed: {e}")
        return text # Fallback to original text on error

def clean_text(text: str) -> str:
    """Basic preprocessing to remove URLs, mentions, and non-alphanumeric characters."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+|\.com\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(@|#)\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text