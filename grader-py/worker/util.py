import os, json, re
import numpy as np
from sentence_transformers import SentenceTransformer

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