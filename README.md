kafka-topics --alter --topic social_posts --partitions 6 --bootstrap-server localhost:9092

$env:PYTHONPATH = $PWD
spark-submit --master local[2] --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 spark_consumer/consumer.py

python kafka_producer/producer.py --topic social_posts --rate 10 --max 3

import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\lowki\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession
import json, time
from pymongo import MongoClient
from models.translator import translate_text, clean_text
from models.sentiment_classifier import classify_text
from langdetect import detect

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "bigdata_project")
MONGO_COLLECTION = os.getenv("MONGO_COLL", "social_posts")

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

    for row in df.collect():
        try:
            item = json.loads(row['value'])
        except Exception as e:
            print(f"JSON decode error: {e}")
            continue

        text = item.get("original_text", "")
        lang = (item.get("language") or "unknown").lower()
        text = clean_text(text).strip()
        if lang == "" or lang == "unknown":
            try:
                lang = detect(text)
            except:
                lang = "unknown"
        translated = text if lang in ["en"] else translate_text(text, lang)

        t0 = time.time()
        pred = classify_text(translated)
        latency_ms = (time.time() - t0) * 1000.0

        # Safe printing with UTF-8
        try:
            print("="*60)
            print("Message:", text)
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
                "original_text": text,
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
