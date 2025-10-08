#!/usr/bin/env python3
"""
Producer that streams the Exorde HF dataset into Kafka topic "social_posts".
Usage:
  python producer_hf.py --topic social_posts --rate 20 --max 10000
Notes:
  - Requires Kafka broker at localhost:9092.
  - Install dependencies from requirements-producer.txt
"""
import argparse, json, time, sys
from kafka import KafkaProducer
from datasets import load_dataset

def make_producer(bootstrap='localhost:9092'):
    return KafkaProducer(bootstrap_servers=[bootstrap],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def stream_exorde(producer, topic, rate, max_items=None):
    ds = load_dataset("Exorde/exorde-social-media-december-2024-week1", split="train")
    print("Loaded dataset size:", len(ds))
    count = 0
    for item in ds:
        payload = {
            "date": item.get("date"),
            "original_text": item.get("original_text") or item.get("text") or "",
            "language": item.get("language") or "",
            "url": item.get("url") or "",
            "author_hash": item.get("author_hash") or "",
            "primary_theme": item.get("primary_theme") or "",
            "english_keywords": item.get("english_keywords") or "",
            "sentiment": item.get("sentiment")  # can be None if not present
        }
        # Print debug
        print("=" * 60)
        print("🌍 Message:", item.get("original_text") or item.get("text") or "")
        print("🗣️ Language:", item.get("language") or "")
        print("📊 Sentiment Score:", round(item.get("sentiment"), 3))
        producer.send(topic, value=payload)
        count += 1
        if max_items and count >= max_items:
            break
        if rate > 0:
            time.sleep(1.0/rate)
    producer.flush()
    print("Stream finished, sent:", count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='social_posts')
    parser.add_argument('--bootstrap', default='localhost:9092')
    parser.add_argument('--rate', type=int, default=10, help='messages/sec, 0 = max speed')
    parser.add_argument('--max', type=int, default=None, help='max items to stream (for testing)')
    args = parser.parse_args()

    p = make_producer(args.bootstrap)
    try:
        stream_exorde(p, args.topic, args.rate, args.max)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        p.close()

if __name__ == "__main__":
    main()