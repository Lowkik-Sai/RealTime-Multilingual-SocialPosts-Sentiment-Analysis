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
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.bindAddress", "10.12.115.26") \
    .config("spark.driver.host", "10.12.115.26") \
    .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.streaming.kafka.maxRatePerPartition", "500") \
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
    .trigger(processingTime="5 seconds") \
    .start()

print("--------------------------------------------------------------------------------------------------------------")
print("Listening to Kafka topic:", topic)
print("--------------------------------------------------------------------------------------------------------------")

query.awaitTermination()

# import os
# import json
# import time
# from pymongo import MongoClient
# from pyspark.sql import SparkSession
# from models.translator import translate_text, clean_text
# from models.sentiment_classifier import get_classifier
# import fasttext

# # ------------------------------------------------------------------------------------
# # Environment setup
# # ------------------------------------------------------------------------------------
# os.environ["PYSPARK_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"
# os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"

# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# MONGO_DB = os.getenv("MONGO_DB", "bigdata_project")
# MONGO_COLLECTION = os.getenv("MONGO_COLL", "social_posts")

# FASTTEXT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lid.176.bin')
# _lang_detect_model_cache = None

# # ------------------------------------------------------------------------------------
# # Initialize Spark
# # ------------------------------------------------------------------------------------
# spark = SparkSession.builder \
#     .appName("MultilingualSentimentConsumer") \
#     .master("local[*]") \
#     .config("spark.sql.shuffle.partitions", "4") \
#     .config("spark.driver.bindAddress", "10.12.115.26") \
#     .config("spark.driver.host", "10.12.115.26") \
#     .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8") \
#     .config("spark.streaming.backpressure.enabled", "true") \
#     .config("spark.streaming.kafka.maxRatePerPartition", "500") \
#     .getOrCreate()

# # ------------------------------------------------------------------------------------
# # Helper: Mini-batch processing
# # ------------------------------------------------------------------------------------
# def process_microbatch(df, epoch_id):
#     count = df.count()
#     print(f"\nEpoch {epoch_id}: received {count} rows")
#     if count == 0:
#         return

#     global _lang_detect_model_cache
#     if _lang_detect_model_cache is None:
#         _lang_detect_model_cache = fasttext.load_model(FASTTEXT_MODEL_PATH)
#     lang_model = _lang_detect_model_cache

#     clf = get_classifier()

#     records = []
#     t0_batch = time.time()

#     pandas_df = df.select("value").toPandas()
#     texts_to_classify = []

#     # --- Preprocessing + translation in loop ---
#     for idx, row in pandas_df.iterrows():
#         try:
#             item = json.loads(row["value"])
#         except Exception:
#             continue

#         original_text = item.get("original_text", "")
#         if not original_text.strip():
#             continue

#         cleaned = clean_text(original_text)
#         if not cleaned:
#             continue

#         # Language detection
#         predictions = lang_model.predict(cleaned.replace("\n", " "), k=1)
#         lang = predictions[0][0].replace('__label__', '') if predictions[0] else "unknown"

#         translated = cleaned if lang == 'en' else translate_text(cleaned)
#         texts_to_classify.append({
#             "original": original_text,
#             "translated": translated,
#             "lang": lang,
#             "meta": item
#         })

#     # --------------------------------------------------------------------------------
#     # Batch Sentiment Classification (20–50 per batch)
#     # --------------------------------------------------------------------------------
#     BATCH_SIZE = 32
#     results = []

#     for i in range(0, len(texts_to_classify), BATCH_SIZE):
#         batch = texts_to_classify[i:i+BATCH_SIZE]
#         batch_texts = [t["translated"] for t in batch]

#         try:
#             outputs = clf(batch_texts)
#         except Exception as e:
#             print("Batch classification error:", e)
#             continue

#         for j, out in enumerate(outputs):
#             text_info = batch[j]
#             label = out.get("label", "NEUTRAL")
#             score = float(out.get("score", 0.0))
#             if score < 0.55:
#                 label = "NEUTRAL"

#             doc = {
#                 "date": text_info["meta"].get("date"),
#                 "original_text": text_info["original"],
#                 "detected_language": text_info["lang"],
#                 "translated_text": text_info["translated"],
#                 "predicted_label": label,
#                 "predicted_score": score,
#                 "processing_latency_ms": 0,  # simplified
#                 "ingest_ts": time.time(),
#             }
#             records.append(doc)
#             print("=" * 60)
#             print("Message:", text_info["original"])
#             print("Language:", text_info["lang"])
#             print("Translated:", text_info["translated"][:120])
#             print("Predicted Label:", label)
#             print("Score:", round(score, 3))

#     # --------------------------------------------------------------------------------
#     # Insert into MongoDB
#     # --------------------------------------------------------------------------------
#     if records:
#         try:
#             client = MongoClient(MONGO_URI)
#             coll = client[MONGO_DB][MONGO_COLLECTION]
#             coll.insert_many(records)
#             client.close()
#             print(f"Inserted {len(records)} records into MongoDB in {int((time.time()-t0_batch)*1000)} ms")
#         except Exception as e:
#             print(f"MongoDB insert failed: {e}")

# # ------------------------------------------------------------------------------------
# # Kafka Streaming setup
# # ------------------------------------------------------------------------------------
# kafka_bootstrap = "localhost:9092"
# topic = "social_posts"

# df = spark.readStream \
#     .format("kafka") \
#     .option("kafka.bootstrap.servers", kafka_bootstrap) \
#     .option("subscribe", topic) \
#     .option("startingOffsets", "earliest") \
#     .option("failOnDataLoss", "false") \
#     .load() \
#     .selectExpr("CAST(value AS STRING) as value")

# query = df.writeStream \
#     .foreachBatch(process_microbatch) \
#     .option("checkpointLocation", "file:///tmp/spark_checkpoint/multilingual_sentiment") \
#     .start()

# print("\n" + "-"*100)
# print(f"Listening to Kafka topic: {topic}")
# print("-"*100)

# query.awaitTermination()
