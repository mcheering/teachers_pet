import os
import requests
from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# helpers for parsing the LLM output
from util import parse_model_json, normalize_graded

# ─── App & Config ───────────────────────────────────────────────────────────────
app = Flask("ai_service")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ─── Image Alt‐Text Endpoint ────────────────────────────────────────────────────
print("[ai_service] Loading BLIP model for alt text…")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/ai/alt_text", methods=["POST"])
def ai_alt_text():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    img_file = request.files["file"]
    try:
        img = Image.open(img_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    try:
        inputs = blip_processor(img, return_tensors="pt")
        out    = blip_model.generate(**inputs)
        alt_text = blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print("[ai_service] BLIP error:", e)
        alt_text = "Image for exam question."

    return jsonify({"alt_text": alt_text.strip().capitalize()})

# ─── Rubric‐Based Grading Endpoint ───────────────────────────────────────────────
@app.route("/grade_criterion", methods=["POST"])
def grade_criterion():
    """
    Expects JSON:
      {
        "answer_text": "...",
        "criteria": [
          { "id": 123, "text": "Describe X", "max_points": 2 },
          { "id": 124, "text": "Explain Y",   "max_points": 1 },
          ...
        ]
      }
    Returns:
      { "results": [
          { "rubric_id": 123, "score": 2, "feedback": "Good detail on X" },
          { "rubric_id": 124, "score": 0, "feedback": "Missing Y explanation" },
          ...
        ]
      }
    """
    data = request.get_json()
    answer_text = data["answer_text"]
    criteria    = data["criteria"]

    # build the rubric lines for the prompt
    rubric_lines = "\n".join(
        f"- ({c['id']}) [{c['max_points']} pts] {c['text']}"
        for c in criteria
    )

    system = "You are an AP grader. Use only the rubric to score. Return ONLY valid JSON."
    user   = f"""Student answer:
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
    raw    = resp.json().get("response", "")
    parsed = parse_model_json(raw)
    results = normalize_graded(parsed)

    # clamp scores, validate structure
    clean = []
    for item in results:
        rid = int(item.get("rubric_id", 0))
        sc  = int(round(item.get("score", 0)))
        # find max_points for this rubric_id
        mp = next((c["max_points"] for c in criteria if c["id"] == rid), 0)
        sc = max(0, min(sc, mp))
        fb = item.get("feedback", "").strip()
        clean.append({"rubric_id": rid, "score": sc, "feedback": fb})

    return jsonify({"results": clean})

if __name__ == "__main__":
    # listen on all interfaces so your Rust worker (and curl) can reach it
    app.run(host="0.0.0.0", port=5005)