import os, re
from typing import List, Dict, Any
import fitz
import pdfplumber

def extract_images_from_pdf(exam_id: int, pdf_path: str, conn, UPLOAD_DIR) -> List[Dict[str, Any]]:
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

def attach_images_heuristic(exam_id: int, page_texts: List[str], qs: List[Dict[str,Any]], pdf_imgs: List[Dict[str,Any]], conn, DEBUG_IMG_LINKS=False):
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

def ingest_scoring_pdf(job, conn, UPLOAD_DIR, DEBUG_IMG_LINKS=False):
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

    mc_qs = [q for q in qs if q["is_mc"]]
    frq_qs = [q for q in qs if not q["is_mc"]]

    mc_count = frq_q_count = part_count = 0

    pdf_imgs = extract_images_from_pdf(exam_id, pdf_path, conn, UPLOAD_DIR)

    import fitz
    doc = fitz.open(pdf_path)
    cur = conn.cursor()
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
            PART_HEADER_RE = re.compile(r'(?m)^\s*Part\s+([A-D])\b', re.I)
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

    attach_images_heuristic(exam_id, page_texts, qs, pdf_imgs, conn, DEBUG_IMG_LINKS)