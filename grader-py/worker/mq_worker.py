import json
import traceback
from pdf_ingest import ingest_scoring_pdf
from grading import grade_answer

def on_msg(ch, method, props, body, conn, UPLOAD_DIR, OLLAMA_URL, DEBUG_IMG_LINKS=False):
    print("[worker] got msg:", body[:120])
    job = json.loads(body)
    kind = job.get("kind", "scoring_pdf")
    try:
        if kind in ("scoring_pdf", "exam_pdf", "rubric_pdf"):
            ingest_scoring_pdf(job, conn, UPLOAD_DIR, DEBUG_IMG_LINKS)
        elif kind == "grade_answer":
            grade_answer(job, conn, OLLAMA_URL)
        else:
            print(f"[worker] unknown kind: {kind}")
    except Exception as e:
        print(f"[worker] ERROR: {e}")
        traceback.print_exc()
        conn.rollback()
    finally:
        ch.basic_ack(method.delivery_tag)