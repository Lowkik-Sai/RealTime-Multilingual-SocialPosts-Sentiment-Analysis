# import threading
# import warnings
# import re
# from transformers import pipeline, MarianMTModel, MarianTokenizer


# # Load translation model (generic multilingual to English using MarianMT)
# model_name = "Helsinki-NLP/opus-mt-mul-en"
# translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
# translator_model = MarianMTModel.from_pretrained(model_name)

# # Map a few common language codes to MarianMT models. Add more as needed.
# LANG_MODEL_MAP = {
#     'es': 'Helsinki-NLP/opus-mt-es-en',
#     'fr': 'Helsinki-NLP/opus-mt-fr-en',
#     'de': 'Helsinki-NLP/opus-mt-de-en',
#     'it': 'Helsinki-NLP/opus-mt-it-en',
#     'pt': 'Helsinki-NLP/opus-mt-pt-en',
#     'hi': 'Helsinki-NLP/opus-mt-hi-en',
#     'ar': 'Helsinki-NLP/opus-mt-ar-en',
#     # add more as needed...
# }

# _lock = threading.Lock()
# _pipelines = {}

# def has_translator_for(lang):
#     return lang in LANG_MODEL_MAP

# def get_translator(lang):
#     """Return HF pipeline for translation for given lang (cached)."""
#     if not has_translator_for(lang):
#         return None
#     with _lock:
#         if lang not in _pipelines:
#             model_name = LANG_MODEL_MAP[lang]
#             # Create pipeline (may download model)
#             _pipelines[lang] = pipeline("translation", model=model_name, device=0 if False else -1)
#         return _pipelines[lang]

# def translate_text(text, lang):
#     """Translate text to English if translator is available else return original."""
#     if not text:
#         return text
#     if lang == '' or lang is None or lang == 'en':
#         return text
#     tp = get_translator(lang)
#     if tp is None:
#         # fallback: return original text
#         return text
#     try:
#         out = tp(text, max_length=512)
#         # pipeline returns list of dicts with 'translation_text'
#         return out[0]['translation_text']
#     except Exception as e:
#         warnings.warn(f"Translation failed for lang={lang}: {e}")
#         return text

# def translate_text(text: str) -> str:
#     batch = translator_tokenizer([text], return_tensors="pt", padding=True)
#     gen = translator_model.generate(**batch)
#     translated = translator_tokenizer.decode(gen[0], skip_special_tokens=True)
#     return translated

# def clean_text(text):
#     """Basic preprocessing to remove links, mentions, etc."""
#     text = re.sub(r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '', text)
#     text = re.sub(r'(@|#)\w+', '', text)
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# models/translator.py
import re
import warnings
from transformers import pipeline

# Global cache for the translation pipeline
_translator_pipeline = None

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
        translator = get_translator()
        # The pipeline returns a list with a dictionary
        result = translator(text, max_length=512)
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