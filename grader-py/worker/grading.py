import requests
from .util import parse_model_json, normalize_graded

def grade_answer(job, conn, OLLAMA_URL):
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

    # build payload for our grader-py service
    payload = {
      "answer_text": answer_text,
      "criteria": [
        { "id": rid, "text": desc, "max_points": pts }
        for rid, desc, pts in bullets
      ]
    }
    resp = requests.post("http://localhost:5000/grade_criterion", json=payload, timeout=120)
    resp.raise_for_status()
    graded = resp.json()["results"]
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