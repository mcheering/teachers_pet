#!/usr/bin/env python3
import os, json, re, traceback
from typing import List, Dict, Any
import numpy as np
import pdfplumber
import pika, psycopg
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# --- New: Flask/Transformers for alt text ---
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ========== CONFIGURATION ==========
DEBUG_QBLOCKS   = bool(os.getenv("DEBUG_QBLOCKS"))
DEBUG_IMG_LINKS = bool(os.getenv("DEBUG_IMG_LINKS"))
AMQP_URL   = os.getenv("AMQP_URL",   "amqp://guest:guest@localhost:5672/%2f")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_DSN     = os.getenv("DB_DSN",     f"dbname=apgrader user={os.getenv('USER')}")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# ========== DB / MQ ==========
conn = psycopg.connect(DB_DSN)
conn.autocommit = False
cur  = conn.cursor()
mq_conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
channel = mq_conn.channel()
channel.queue_declare(queue="pdf_ingest")

# ========== Embeddings ==========
_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def to_pgvector(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"

# ========== Robust JSON parse ==========
def parse_model_json(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'($begin:math:display$.*$end:math:display$|\{.*\})', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    print("[grade] Could not parse JSON. First 400 chars:", text[:400])
    return []

def normalize_graded(obj):
    if not obj:
        return []
    if isinstance(obj, list):
        if len(obj) == 1 and isinstance(obj[0], dict):
            for k in ("answers","responses","grades","items","result"):
                if k in obj[0] and isinstance(obj[0][k], list):
                    return obj[0][k]
        if obj and isinstance(obj[0], dict) and {"rubric_id","score","feedback"} <= obj[0].keys():
            return obj
        return []
    if isinstance(obj, dict):
        for k in ("answers","responses","grades","items","result"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    return []

# ========== Image extraction (all images per page) ==========
def extract_images_from_pdf(exam_id: int, pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    cur2 = conn.cursor()
    out_dir = os.path.join(UPLOAD_DIR, "img")
    os.makedirs(out_dir, exist_ok=True)
    imgs: List[Dict[str, Any]] = []
    for pno in range(len(doc)):
        page = doc[pno]
        for xref, *_ in page.get_images(full=True):
            try:
                rect = page.get_image_bbox(xref)
            except Exception:
                rect = fitz.Rect(0, 0, 0, 0)
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fname = f"{exam_id}_p{pno}_{xref}.png"
            fpath = os.path.join(out_dir, fname)
            pix.save(fpath)
            cur2.execute(
                """INSERT INTO pdf_images (exam_id,page,x0,y0,x1,y1,width,height,file_path)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id""",
                (exam_id, pno, rect.x0, rect.y0, rect.x1, rect.y1, pix.width, pix.height, fpath)
            )
            img_id = cur2.fetchone()[0]
            imgs.append({
                "id": img_id,
                "page": pno,
                "x0": rect.x0, "y0": rect.y0,
                "x1": rect.x1, "y1": rect.y1,
                "width": pix.width, "height": pix.height,
                "file_path": fpath,
            })
    conn.commit()
    return imgs

def attach_images_heuristic(exam_id: int, page_texts: List[str], qs: List[Dict[str,Any]], pdf_imgs: List[Dict[str,Any]]):
    cur3 = conn.cursor()
    starts: Dict[int, int] = {}
    qnums = [q["qnum"] for q in qs]
    for idx, qnum in enumerate(qnums):
        pat = re.compile(rf"(?m)^\s*(?:Question\s+)?{qnum}[.)]\s")
        for pno, txt in enumerate(page_texts):
            if pat.search(txt):
                starts[qnum] = pno
                break
        else:
            starts[qnum] = starts.get(qnums[idx-1], 0) if idx > 0 else 0

    spans: Dict[int, tuple] = {}
    for i, qnum in enumerate(qnums):
        start = starts[qnum]
        end   = starts.get(qnums[i+1], len(page_texts)) if i+1 < len(qnums) else len(page_texts)
        spans[qnum] = (start, end)

    def link_question(qid, img_ids):
        for iid in img_ids:
            cur3.execute(
                "INSERT INTO question_images (question_id,image_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (qid, iid)
            )

    def link_part(pid, img_ids):
        for iid in img_ids:
            cur3.execute(
                "INSERT INTO frq_part_images (frq_part_id,image_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (pid, iid)
            )

    for q in qs:
        qnum = q["qnum"]
        start, end = spans[qnum]
        imgs_here = [im["id"] for im in pdf_imgs if start <= im["page"] < end]
        if not imgs_here:
            continue

        if q.get("is_mc", 1 if 1 <= qnum <= 10 else 0):
            cur3.execute("SELECT id FROM mc_questions WHERE exam_id=%s AND q_number=%s", (exam_id, qnum))
            row = cur3.fetchone()
            if row:
                link_question(row[0], imgs_here)
        else:
            cur3.execute("""
                SELECT fp.id
                FROM frq_parts fp
                JOIN questions q ON fp.question_id=q.id
                WHERE q.exam_id=%s AND q.q_number=%s
            """, (exam_id, qnum))
            for (pid,) in cur3.fetchall():
                link_part(pid, imgs_here)

    if DEBUG_IMG_LINKS:
        print("[img] linked question_images:", exam_id)
        cur3.execute("SELECT * FROM question_images WHERE question_id IN (SELECT id FROM questions WHERE exam_id=%s)", (exam_id,))
        print(cur3.fetchall())
        cur3.execute("SELECT * FROM frq_part_images WHERE frq_part_id IN (SELECT fp.id FROM frq_parts fp JOIN questions q ON fp.question_id=q.id WHERE q.exam_id=%s)", (exam_id,))
        print(cur3.fetchall())
    conn.commit()
    print("[scoring_pdf] attached images via heuristic")

# ========== Scoring PDF ingest ==========
def ingest_scoring_pdf(job):
    exam_id, pdf_path = job["exam_id"], job["path"]
    print(f"[scoring_pdf] {pdf_path}")

    # page texts & whole text
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [p.extract_text() or "" for p in pdf.pages]
    full = "\n".join(page_texts)

    # ---------- regexes ----------
    QNUM_RE   = re.compile(r'(?m)^\s*(?:Question\s+)?(\d{1,3})[.)]\s')
    CHOICE_RE = re.compile(r'^\s*$begin:math:text$[A-E]$end:math:text$', re.MULTILINE)
    PART_HEADER_RE = re.compile(r'(?m)^\s*Part\s+([A-D])\b', re.I)

    # Split questions by number, skipping intro text before Q1
    parts = QNUM_RE.split(full)
    qs: List[Dict[str, Any]] = []
    for i in range(1, len(parts), 2):
        qnum = int(parts[i])
        block = parts[i+1].strip()
        qs.append({"qnum": qnum, "block": block, "is_mc": 1 if 1 <= qnum <= 10 else 0})

    # --- Partition MC/FRQ based on Q number ---
    mc_qs = [q for q in qs if q["is_mc"]]
    frq_qs = [q for q in qs if not q["is_mc"]]

    mc_count = frq_q_count = part_count = 0

    # ---- IMAGE EXTRACTION (grabs all images and their bounding boxes) ----
    pdf_imgs = extract_images_from_pdf(exam_id, pdf_path)

    # ---------- insert MC ----------
    doc = fitz.open(pdf_path)
    for q in mc_qs:
        qnum, block = q["qnum"], q["block"]
        stem_end = block.find("(A)")
        stem = block[:stem_end].strip() if stem_end != -1 else block
        cur.execute(
            "INSERT INTO mc_questions (exam_id,q_number,stem) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",
            (exam_id, qnum, stem)
        )
        mc_id = cur.fetchone()[0]
        mc_count += 1

        # Split choices by "(A)", "(B)", etc.
        pieces = CHOICE_RE.split(block)
        choices = CHOICE_RE.findall(block)
        q_page = None
        for page_num, page in enumerate(doc):
            if str(qnum) in (page.get_text() or ""):
                q_page = page_num
                break
        if q_page is None:
            q_page = 0
        page = doc[q_page]
        text_blocks = page.get_text("blocks")

        # --- find label blocks (choices) ---
        label_blocks = []
        for idx, label in enumerate(choices):
            lbl = f"({chr(65 + idx)})"
            found = False
            for b in text_blocks:
                if lbl in b[4]:
                    label_blocks.append({
                        "label": lbl,
                        "rect": (b[0], b[1], b[2], b[3]),
                        "y_mid": (b[1] + b[3]) / 2,
                        "idx": idx
                    })
                    found = True
                    break
            if not found:
                label_blocks.append(None)

        # --- find images on this page ---
        page_imgs = [img for img in pdf_imgs if img['page'] == q_page]
        linked_images = set()
        for idx, lblock in enumerate(label_blocks):
            if lblock is None:
                continue
            best_img = None
            best_dist = 1e6
            for img in page_imgs:
                y_img_mid = (img['y0'] + img['y1']) / 2
                dist = abs(lblock["y_mid"] - y_img_mid)
                if dist < best_dist and 0 < (lblock["y_mid"] - y_img_mid) < 120:
                    best_dist = dist
                    best_img = img
            cur.execute(
                "INSERT INTO mc_choices (mc_question_id, choice_label, choice_text) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",
                (mc_id, chr(65 + idx), pieces[idx + 1].strip() if idx + 1 < len(pieces) else "")
            )
            mc_choice_id = cur.fetchone()[0]
            if best_img and best_img['id'] not in linked_images:
                cur.execute(
                    "INSERT INTO mc_choice_images (mc_choice_id, image_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                    (mc_choice_id, best_img['id'])
                )
                linked_images.add(best_img['id'])

    # ---------- insert FRQ ----------
    for q in frq_qs:
        qnum, block = q["qnum"], q["block"]
        cur.execute(
            "INSERT INTO questions (exam_id,q_number,q_type,prompt) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",
            (exam_id, qnum, 'FRQ', block)
        )
        question_id = cur.fetchone()[0]
        frq_q_count += 1

        if qnum == 11:
            cur.execute(
                "INSERT INTO frq_parts (question_id, part_label, prompt_text) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (question_id, 'a', block)
            )
            part_count += 1
        elif qnum == 12:
            parts_ = list(PART_HEADER_RE.finditer(block))
            for idx, m in enumerate(parts_):
                label = m.group(1).lower()
                start = m.end()
                end = parts_[idx + 1].start() if idx + 1 < len(parts_) else len(block)
                txt = block[start:end].strip()
                cur.execute(
                    "INSERT INTO frq_parts (question_id, part_label, prompt_text) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (question_id, label, txt)
                )
                part_count += 1

    conn.commit()
    print(f"[scoring_pdf] Qs stored: MC={mc_count}, FRQ_Q={frq_q_count}, FRQ_parts={part_count}")

# ========== Grading (unchanged) ==========
def grade_answer(job):
    answer_id = job["answer_id"]
    print(f"[grade] answer {answer_id}")

    cur2 = conn.cursor()
    cur2.execute("""
        SELECT a.answer_text, a.frq_part_id, fp.part_label, q.q_number
        FROM answers a
        LEFT JOIN frq_parts fp ON a.frq_part_id = fp.id
        LEFT JOIN questions q  ON a.question_id = q.id
        WHERE a.id = %s
    """, (answer_id,))
    row = cur2.fetchone()
    if not row:
        print(f"[grade] no answer {answer_id}")
        return
    answer_text, frq_part_id, part_label, qnum = row

    cur2.execute("""
        SELECT id, description, max_points
        FROM rubric_criteria
        WHERE frq_part_id = %s
        ORDER BY order_index
    """, (frq_part_id,))
    bullets = cur2.fetchall()
    if not bullets:
        print(f"[grade] no rubric bullets for part {frq_part_id}")
        return

    rubric_lines = "\n".join(f"- ({rid}) [{pts} pts] {desc}" for rid, desc, pts in bullets)

    system = "You are an AP grader. Use only the rubric to score. Return ONLY valid JSON."
    user = f"""Question {qnum} Part {part_label}

Student answer:
\"\"\"{answer_text}\"\"\"

Rubric bullets (id in parens):
{rubric_lines}

Return a JSON array like:
[{{"rubric_id": 123, "score": 1, "feedback": "specific feedback"}}]"""

    payload = {
        "model": "llama3:8b",
        "prompt": f"<|system|>{system}\n<|user|>{user}\n<|assistant|>",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    print("[grade] ollama status:", resp.status_code)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    print("[grade] raw first 200:", raw[:200].replace("\n"," "))

    parsed = parse_model_json(raw)
    graded = normalize_graded(parsed)
    print("[grade] parsed/normalized:", graded)

    if not graded:
        print("[grade] empty/invalid graded array, aborting")
        conn.rollback()
        return

    clean = []
    for i, it in enumerate(graded):
        try:
            rid = int(it["rubric_id"])
            sc  = float(it.get("score", 0))
            fb  = str(it.get("feedback", ""))
            clean.append({"rubric_id": rid, "score": sc, "feedback": fb})
        except Exception as e:
            print(f"[grade] bad item {i} -> {it}: {e}")

    if not clean:
        print("[grade] nothing to store")
        conn.rollback()
        return

    total = int(round(sum(i["score"] for i in clean)))
    cur2.execute("""
        INSERT INTO grade_runs (answer_id, model_name, total_score)
        VALUES (%s,%s,%s)
        RETURNING id
    """, (answer_id, "llama3:8b", total))
    run_id = cur2.fetchone()[0]

    for item in clean:
        cur2.execute("""
            INSERT INTO grade_items (grade_run_id, rubric_criteria_id, score, feedback)
            VALUES (%s,%s,%s,%s)
        """, (run_id, item["rubric_id"], item["score"], item["feedback"]))

    try:
        cur2.execute(
            "UPDATE frq_submissions SET graded=true, latest_grade_run_id=%s WHERE answer_id=%s",
            (run_id, answer_id)
        )
    except Exception as e:
        print(f"[grade] WARN could not update frq_submissions: {e}")

    conn.commit()
    print(f"[grade] stored run {run_id}")


# ========== AI Alt Text Flask Endpoint ==========
app = Flask("ai_alt_text")

print("[ai_alt_text] Loading BLIP model for alt text...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/ai/alt_text", methods=["POST"])
def ai_alt_text():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    img_file = request.files["file"]
    try:
        img = Image.open(img_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    # Generate alt text
    try:
        inputs = blip_processor(img, return_tensors="pt")
        out = blip_model.generate(**inputs)
        alt_text = blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print("[ai_alt_text] Model error:", e)
        alt_text = "Image for exam question."

    # Optionally clean up alt_text
    alt_text = alt_text.strip().capitalize()
    return jsonify({"alt_text": alt_text})

# ========== MQ callback ==========
def on_msg(ch, method, props, body):
    print("[worker] got msg:", body[:120])
    job = json.loads(body)
    kind = job.get("kind", "scoring_pdf")
    try:
        if kind in ("scoring_pdf", "exam_pdf", "rubric_pdf"):
            ingest_scoring_pdf(job)
        elif kind == "grade_answer":
            grade_answer(job)
        else:
            print(f"[worker] unknown kind: {kind}")
    except Exception as e:
        print(f"[worker] ERROR: {e}")
        traceback.print_exc()
        conn.rollback()
    finally:
        ch.basic_ack(method.delivery_tag)
        
@app.route("/grade_criterion", methods=["POST"])
def grade_criterion():
    """
    Expects JSON:
      {
        "answer_text": "...",
        "criteria": [
          { "id": 123, "text": "Describe X", "max_points": 2 },
          ...
        ]
      }
    Returns JSON:
      { "results": [
          { "rubric_id": 123, "score": 2, "feedback": "…" }, …
        ]
      }
    """
    data = request.get_json()
    answer_text = data["answer_text"]
    criteria    = data["criteria"]

    # build the rubric lines exactly as your worker_frq.rs expects
    rubric_lines = "\n".join(
        f"- ({c['id']}) [{c['max_points']} pts] {c['text']}"
        for c in criteria
    )

    system = "You are an AP grader. Use only the rubric to score. Return ONLY valid JSON."
    user = f"""Student answer:
\"\"\"{answer_text}\"\"\"

Rubric bullets (id in parens):
{rubric_lines}

Return a JSON array like:
[{{"rubric_id": 123, "score": 1, "feedback": "specific feedback"}}]"""

    llama_payload = {
        "model":   "llama3:8b",
        "prompt":  f"<|system|>{system}\n<|user|>{user}\n<|assistant|>",
        "format":  "json",
        "stream":  False,
        "options": {"temperature": 0.0}
    }

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=llama_payload, timeout=120)
    resp.raise_for_status()
    raw     = resp.json().get("response","")
    parsed  = parse_model_json(raw)
    results = normalize_graded(parsed)

    return jsonify({"results": results})

# ========== START Flask in a thread and worker as main ==========
if __name__ == "__main__":
    import threading
    def run_flask():
        app.run(host="0.0.0.0", port=5005, threaded=True)
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    print("[worker] waiting for jobs")
    channel.basic_consume(queue="pdf_ingest", on_message_callback=on_msg)
    channel.start_consuming()