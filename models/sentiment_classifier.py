import threading
from transformers import pipeline

_lock = threading.Lock()
_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )

def get_classifier():
    global _classifier
    with _lock:
        if _classifier is None:
            # Pretrained for social media sentiment
            _classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
        return _classifier

def classify_text(text: str):
    """
    Classifies English text as Positive, Negative, or Neutral.
    Returns a dict with label and score.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"label": "NEUTRAL", "score": 0.0}
    
    # clf = get_classifier()
    clf = _classifier
    try:
        result = clf(text[:512])[0]
        return {"label": result["label"], "score": float(result["score"])}
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.0}


# models/sentiment_classifier.py
# from transformers import pipeline

# # Global cache for the classifier pipeline
# _classifier_pipeline = None
# _classifier_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# _label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

# def get_classifier():
#     """Lazily loads and caches the sentiment analysis model."""
#     global _classifier_pipeline
#     if _classifier_pipeline is None:
#         _classifier_pipeline = pipeline(
#             "sentiment-analysis",
#             model=_classifier_model_name,
#             device=-1
#         )
#     return _classifier_pipeline

# def classify_text(text: str) -> dict:
#     """
#     Classifies text using a Twitter-trained RoBERTa model.
#     Returns a dictionary with 'label' and 'score'.
#     """
#     if not text or not isinstance(text, str):
#         return {"label": "Neutral", "score": 0.0}
#     try:
#         classifier = get_classifier()
#         # Truncate text to fit model's max input size
#         result = classifier(text, truncation=True, max_length=512)[0]
#         return {
#             "label": _label_mapping.get(result['label'], 'Neutral'),
#             "score": float(result['score'])
#         }
#     except Exception:
#         return {"label": "Neutral", "score": 0.0}