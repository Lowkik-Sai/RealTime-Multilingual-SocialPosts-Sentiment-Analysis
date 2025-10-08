import json
from django.shortcuts import render
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "bigdata_project")
MONGO_COLLECTION = os.getenv("MONGO_COLL", "social_posts")

def home(request):
    client = MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_COLLECTION]

    pipeline = [
        {"$group": {"_id": {"lang": "$detected_language", "label": "$predicted_label"}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = list(coll.aggregate(pipeline))

    data = {}
    for r in results:
        lang = r["_id"]["lang"]
        label = r["_id"]["label"]
        if lang not in data:
            data[lang] = {}
        data[lang][label] = r["count"]

    # convert to JSON string
    data_json = json.dumps(data)

    return render(request, "dashboard.html", {"sentiment_data": data_json})
