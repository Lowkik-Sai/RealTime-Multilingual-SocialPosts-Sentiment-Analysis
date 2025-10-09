import json
from django.shortcuts import render
from pymongo import MongoClient
import os
from django.http import JsonResponse

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "bigdata_project")
MONGO_COLLECTION = os.getenv("MONGO_COLL", "social_posts")

# def per_language_sentiment(request):
#     client = MongoClient(MONGO_URI)
#     coll = client[MONGO_DB][MONGO_COLLECTION]

#     # Aggregate per language per sentiment
#     pipeline = [
#         {"$group": {"_id": {"lang": "$detected_language", "label": "$predicted_label"}, "count": {"$sum": 1}}},
#         {"$sort": {"count": -1}}
#     ]
#     results = list(coll.aggregate(pipeline))

#     data = {}
#     for r in results:
#         lang = r["_id"]["lang"]
#         label = r["_id"]["label"]
#         if lang not in data:
#             data[lang] = {}
#         data[lang][label] = r["count"]

#     return render(request, "per_language_sentiment.html", {"sentiment_data": json.dumps(data)})

# def live_feed(request):
#     client = MongoClient(MONGO_URI)
#     coll = client[MONGO_DB][MONGO_COLLECTION]

#     latest_posts = list(coll.find().sort("ingest_ts", -1).limit(50))
#     return render(request, "live_feed.html", {"posts": latest_posts})

POSTS_PER_PAGE = 50

client = MongoClient(MONGO_URI)
coll = client[MONGO_DB][MONGO_COLLECTION]

def live_feed(request):
    return render(request, "live_feed.html")

def live_feed_data(request):
    """
    Returns latest posts in JSON format with pagination
    """
    page = int(request.GET.get("page", 1))
    skip = (page - 1) * POSTS_PER_PAGE

    total_posts = coll.count_documents({})
    total_pages = (total_posts + POSTS_PER_PAGE - 1) // POSTS_PER_PAGE

    cursor = coll.find().sort("ingest_ts", -1).skip(skip).limit(POSTS_PER_PAGE)
    posts = []
    for post in cursor:
        posts.append({
            "date": post.get("date"),
            "original_text": post.get("original_text"),
            "translated_text": post.get("translated_text"),
            "predicted_label": post.get("predicted_label"),
            "predicted_score": post.get("predicted_score")
        })

    return JsonResponse({"posts": posts, "total_pages": total_pages})

TOP_LANGUAGES = 10 

def per_language_sentiment(request):
    """
    Renders the page
    """
    return render(request, "per_language_sentiment.html")


def per_language_data(request):
    """
    Returns aggregated sentiment counts per language as JSON
    """
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

    # Sort languages by total posts and take top N
    top_langs = sorted(data.keys(), key=lambda x: sum(data[x].values()), reverse=True)[:TOP_LANGUAGES]
    filtered_data = {lang: data[lang] for lang in top_langs}

    return JsonResponse(filtered_data)