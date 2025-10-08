# import os
# os.environ["PYSPARK_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"
# os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"

# import re
# import json
# import time
# from kafka import KafkaConsumer
# from pymongo import MongoClient
# from transformers import pipeline, MarianMTModel, MarianTokenizer
# from langdetect import detect

# # from deep_translator import GoogleTranslator

# # text_to_translate = "Ceci est un exemple de texte en français."
# # translated_text = GoogleTranslator(source='auto', target='en').translate(text_to_translate)
# # print(f"Original: {text_to_translate}")
# # print(f"Translated (French): {translated_text}")


# # MongoDB connection
# client = MongoClient("localhost", 27017)
# db = client["bigdata_project"]
# collection = db["social_posts"]

# # Load sentiment classifier (English)
# sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# # Load translation model (generic multilingual to English using MarianMT)
# model_name = "Helsinki-NLP/opus-mt-mul-en"
# translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
# translator_model = MarianMTModel.from_pretrained(model_name)

# def translate_text(text: str, lang: str) -> str:
#     """Translate text into English if not English"""
#     if lang == "en":
#         return text
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

# # Kafka consumer
# consumer = KafkaConsumer(
#     "social_posts",
#     bootstrap_servers=["localhost:9092"],
#     auto_offset_reset="earliest",
#     enable_auto_commit=True,
#     group_id="my-group",
#     value_deserializer=lambda x: json.loads(x.decode("utf-8"))
# )

# print("🚀 Listening for messages on Kafka topic: social_posts")

# for message in consumer:
#     try:
#         item = message.value
#         text = item.get("original_text", "")
#         lang = (item.get("language") or "unknown").lower()

#         # Step 1: clean
#         cleaned = clean_text(text)

#         # Step 2: detect language (fallback if not provided)
#         if lang == "" or lang == "unknown":
#             try:
#                 lang = detect(cleaned)
#             except:
#                 lang = "unknown"

#         # Step 3: translate if needed
#         if lang != "en" and cleaned.strip():
#             translated = translate_text(cleaned, lang)
#         else:
#             translated = cleaned

#         # Step 4: classify
#         if translated.strip():
#             pred = sentiment_pipeline(translated[:512])[0]  # truncate to avoid long inputs
#             label = pred["label"]
#             score = float(pred["score"])
#         else:
#             label = "neutral"
#             score = 0.0

#         # Step 5: prepare document
#         tweet_doc = {
#             "date": item.get("date"),
#             "url": item.get("url"),
#             "author_hash": item.get("author_hash"),
#             "original_text": text,
#             "detected_language": lang,
#             "translated_text": translated,
#             "predicted_label": label,
#             "predicted_score": score,
#             "ts": time.time()
#         }

#         # Insert into MongoDB
#         collection.insert_one(tweet_doc)

#         # Print debug
#         print("=" * 60)
#         print("🌍 Original:", text)
#         print("🗣️ Detected Lang:", lang)
#         print("🔤 Translated:", translated)
#         print("📊 Sentiment:", label, "(score:", round(score, 3), ")")

#     except Exception as e:
#         print("❌ Error processing message:", str(e))

import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession
import json, time
from pymongo import MongoClient
from models.translator import translate_text, clean_text
from models.sentiment_classifier import classify_text
from langdetect import detect
import fasttext

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "bigdata_project")
MONGO_COLLECTION = os.getenv("MONGO_COLL", "social_posts")

FASTTEXT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lid.176.bin')

# This will act as a cache on each worker node.
_lang_detect_model_cache = None

# Spark session for local Windows
spark = SparkSession.builder \
    .appName("MultilingualSentimentConsumer") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.driver.bindAddress", "10.12.115.26") \
    .config("spark.driver.host", "10.12.115.26") \
    .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8") \
    .getOrCreate()

def process_microbatch(df, epoch_id):
    count = df.count()
    print(f"\nEpoch {epoch_id}: received {count} rows")
    if count == 0:
        return

    global _lang_detect_model_cache
    # ADDED: Lazy-loading of the model. It loads ONCE per worker process.
    if _lang_detect_model_cache is None:
        _lang_detect_model_cache = fasttext.load_model(FASTTEXT_MODEL_PATH)
    
    lang_model = _lang_detect_model_cache

    for row in df.collect():
        try:
            item = json.loads(row['value'])
        except Exception as e:
            print(f"JSON decode error: {e}")
            continue
        
        originalText = item.get("original_text", "")
        text = originalText
        lang = (item.get("language") or "unknown").lower()

        t0 = time.time()

        # Step 1: clean
        text = clean_text(text)        
        # Detect language
        predictions = lang_model.predict(text.replace("\n", " "), k=1)
        lang = predictions[0][0].replace('__label__', '') if predictions[0] else "unknown"

        translated = text if lang == 'en' else translate_text(text)
        pred = classify_text(translated)

        latency_ms = (time.time() - t0) * 1000.0

        # Safe printing with UTF-8
        try:
            print("="*60)
            print("Message:", originalText)
            print("Language:", lang)
            print("Sentiment Score:", round(item.get("sentiment") or 0, 3))
            print("Translated:", translated)
            print("Predicted Label:", pred.get("label"))
            print("Score:", round(pred.get("score"), 3))
            print("Processing latency (ms):", round(latency_ms, 2))
        except UnicodeEncodeError:
            print("UTF-8 encoding issue while printing message")

        # Insert into MongoDB
        try:
            client = MongoClient(MONGO_URI)
            coll = client[MONGO_DB][MONGO_COLLECTION]
            coll.insert_one({
                "date": item.get("date"),
                "original_text": originalText,
                "detected_language": lang,
                "translated_text": translated,
                "predicted_label": pred.get("label"),
                "predicted_score": pred.get("score"),
                "processing_latency_ms": latency_ms,
                "ingest_ts": time.time()
            })
            client.close()
            print("Inserted record into MongoDB")
        except Exception as e:
            print(f"MongoDB insert failed: {e}")

kafka_bootstrap = "localhost:9092"
topic = "social_posts"

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load() \
    .selectExpr("CAST(value AS STRING) as value")

query = df.writeStream \
    .foreachBatch(process_microbatch) \
    .option("checkpointLocation", "file:///tmp/spark_checkpoint/multilingual_sentiment") \
    .start()

print("--------------------------------------------------------------------------------------------------------------")
print("Listening to Kafka topic:", topic)
print("--------------------------------------------------------------------------------------------------------------")

query.awaitTermination()