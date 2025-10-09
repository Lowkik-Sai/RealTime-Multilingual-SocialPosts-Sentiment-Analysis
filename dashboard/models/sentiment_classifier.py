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
