#!/usr/bin/env python3
import os
import pika
import psycopg

from ai_service import app as ai_app
from mq_worker import on_msg

# ========== CONFIGURATION ==========
DEBUG_QBLOCKS   = bool(os.getenv("DEBUG_QBLOCKS"))
DEBUG_IMG_LINKS = bool(os.getenv("DEBUG_IMG_LINKS"))
AMQP_URL   = os.getenv("AMQP_URL",   "amqp://guest:guest@localhost:5672/%2f")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_DSN     = os.getenv("DB_DSN",     f"dbname=apgrader user={os.getenv('USER')}")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

conn = psycopg.connect(DB_DSN)
conn.autocommit = False

mq_conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
channel = mq_conn.channel()
channel.queue_declare(queue="pdf_ingest")

import threading
def run_flask():
    ai_app.run(host="0.0.0.0", port=5005, threaded=True)
t = threading.Thread(target=run_flask, daemon=True)
t.start()
print("[worker] waiting for jobs")
channel.basic_consume(queue="pdf_ingest",
    on_message_callback=lambda ch, method, props, body:
        on_msg(ch, method, props, body, conn, UPLOAD_DIR, OLLAMA_URL, DEBUG_IMG_LINKS)
)
channel.start_consuming()