# answer_builder.py
# ---------------------------------------------------------
# Day 82: Quote-First Answer Builder (No LLM, deterministic)
# (Upgrades Day 75 bullets → ranked quotes + short answer)
# ---------------------------------------------------------

from typing import List, Dict, Any
import re


FALLBACK_MESSAGE = (
    "I couldn't find strong enough evidence in the documents "
    "to answer confidently."
)

MAX_CHUNKS = 4          # scan a bit more than before
MAX_CANDIDATES = 40     # max bullet/sentence candidates scored
MAX_QUOTES = 4          # quotes to show
MAX_BULLET_LEN = 200    # cap per quote line

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "being", "by", "as", "at", "from", "that", "this", "it",
    "its", "they", "them", "you", "your", "we", "our", "i", "me", "my", "can", "may",
    "must", "should", "will", "would", "about", "tell", "explain", "what", "define", "meaning"
}


def build_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    trust_gate: bool = False,
) -> Dict[str, Any]:
    if not chunks:
        return _low_confidence(status="no_evidence")

    # order by score (still useful)
    chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    # If not trusting gate, keep a tiny fallback safety.
    if not trust_gate:
        top_score = chunks[0].get("score", 0.0)
        # NOTE: reranker logits can be negative; so only apply this safety when NOT trusting gate.
        if top_score is None:
            return _low_confidence(status="low_confidence")
        try:
            if float(top_score) <= 0:
                return _low_confidence(status="low_confidence")
        except Exception:
            return _low_confidence(status="low_confidence")

    q_terms = _query_terms(query)
    if not q_terms:
        # if query is too generic, fall back to your old bullet behavior
        return _bullets_fallback(query, chunks)

    # 1) Normalize & dedupe top chunks
    norm = _normalize_chunks(chunks[: max(MAX_CHUNKS, 1)])
    deduped = _dedupe_chunks(norm)

    # 2) Extract bullet/sentence candidates and score them
    candidates = []
    scanned_chunks = 0

    for c in deduped:
        scanned_chunks += 1
        src = c.get("source", "unknown source")
        txt = c.get("text", "")
        lines = _to_lines(txt)  # bullets if present else sentences
        for ln in lines:
            ln2 = _compress_line(ln)
            if not ln2:
                continue
            score = _line_score(ln2, q_terms)
            if score <= 0:
                continue
            candidates.append((score, ln2, src))

            if len(candidates) >= MAX_CANDIDATES:
                break
        if len(candidates) >= MAX_CANDIDATES:
            break

    if not candidates:
        return {
            "answer": "I found documents, but none contain a clear line that matches your question.",
            "confidence": 0.2 if trust_gate else 0.1,
            "used_sources": [],
            "quotes": [],
            "status": "weak_evidence",
            "debug": {"scanned_chunks": scanned_chunks, "picked_quotes": 0},
        }

    # 3) Rank + pick top quotes, limiting to 1–2 sources
    candidates.sort(key=lambda x: x[0], reverse=True)

    picked = []
    used_sources = []
    seen_keys = set()

    for score, text, src in candidates:
        key = _norm_key(text)
        if not key or key in seen_keys:
            continue

        # limit source spread to 2
        if src not in used_sources and len(used_sources) >= 2:
            continue

        seen_keys.add(key)
        picked.append({"source": src, "text": text, "score": round(score, 4)})

        if src not in used_sources:
            used_sources.append(src)

        if len(picked) >= MAX_QUOTES:
            break

    if not picked:
        return _low_confidence(status="weak_evidence")

    # 4) Build a short answer from top 1–3 picked lines
    answer = _make_short_answer([p["text"] for p in picked])

    # 5) Confidence heuristic: monotonic in top quote score
    top_line_score = float(picked[0]["score"])
    conf = min(1.0, 0.4 + top_line_score)  # simple, deterministic
    if not trust_gate:
        conf = min(conf, 0.6)

    return {
        "answer": answer,
        "confidence": round(conf, 3),
        "used_sources": used_sources,
        "quotes": [{"source": p["source"], "text": p["text"]} for p in picked],
        "status": "ok",
        "debug": {
            "scanned_chunks": scanned_chunks,
            "picked_quotes": len(picked),
            "top_quote_score": round(top_line_score, 4),
            "q_terms": q_terms[:12],
        },
    }


# ----------------------------
# Day 82 helpers (query terms + scoring)
# ----------------------------

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _query_terms(query: str) -> List[str]:
    toks = _normalize(query).split()
    terms = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        terms.append(t)
    # de-dup preserve order
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _line_score(line: str, q_terms: List[str]) -> float:
    if not line or not q_terms:
        return 0.0

    s_norm = _normalize(line)
    tokens = set(s_norm.split())

    overlap = sum(1 for t in q_terms if t in tokens)
    base = overlap / max(1, len(q_terms))

    # small boosts
    if any(ch.isdigit() for ch in line):
        base += 0.05
    if any(w in tokens for w in ("probation", "eligibility", "exception", "leave", "attendance", "salary", "notice")):
        base += 0.05

    # penalty: overly generic lines
    if len(tokens) < 6:
        base -= 0.05

    return max(0.0, base)

def _make_short_answer(lines: List[str]) -> str:
    # pick 1–3 lines and stitch; keep it short and readable
    kept = []
    for ln in lines[:3]:
        x = ln.strip()
        if not x:
            continue
        # ensure sentence ends nicely
        if x[-1] not in ".!?":
            x += "."
        kept.append(x)
    return " ".join(kept).strip()


# ----------------------------
# Your original helper family (kept)
# ----------------------------

def _normalize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        source = c.get("source") or "unknown source"
        score = c.get("score", 0.0)
        out.append({"text": text, "source": source, "score": score})
    return out

def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        key = _norm_key(c.get("text", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out

def _to_lines(text: str) -> List[str]:
    """
    Deterministic line extraction:
    - If the chunk already contains numbered/bulleted lines, keep them
    - Else split into sentences
    """
    t = (text or "").strip()
    if not t:
        return []

    lines = [ln.strip(" -•\t") for ln in t.splitlines() if ln.strip()]

    # detect bullet lines by prefix
    bullet_pref = []
    for ln in lines:
        if re.match(r"^(\d+[\).]|•|-)\s*", ln):
            bullet_pref.append(ln)

    # if chunk looks like a list, treat each line as a candidate
    if len(bullet_pref) >= 2 or len(lines) >= 3:
        out = []
        for ln in lines:
            ln2 = re.sub(r"^(\d+[\).]|•|-)\s*", "", ln).strip()
            if ln2:
                out.append(ln2)
        return out[:12]

    # else sentence split
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s.strip() for s in sents if s.strip()]
    return sents[:8]

def _compress_line(line: str) -> str:
    x = " ".join((line or "").split())
    if not x:
        return ""
    x = re.sub(r"\s{2,}", " ", x).strip()
    if len(x) > MAX_BULLET_LEN:
        x = x[: MAX_BULLET_LEN - 3].rstrip() + "..."
    return x

def _norm_key(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s]", "", t)
    return t

def _low_confidence(status: str = "low_confidence") -> Dict[str, Any]:
    return {
        "answer": FALLBACK_MESSAGE,
        "confidence": 0.0,
        "used_sources": [],
        "quotes": [],
        "status": status,
        "debug": {},
    }

def _bullets_fallback(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If query terms are too weak, revert to your grouped-by-source bullets behavior.
    (Still deterministic, still safe.)
    """
    # reuse your earlier behavior but in Day-82 return shape
    chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
    norm = _normalize_chunks(chunks[: max(MAX_CHUNKS, 1)])
    deduped = _dedupe_chunks(norm)

    bullets = []
    used_sources = []
    for c in deduped:
        src = c.get("source", "unknown source")
        txt = c.get("text", "")
        lines = _to_lines(txt)
        for ln in lines[:6]:
            ln2 = _compress_line(ln)
            if ln2:
                bullets.append({"source": src, "text": ln2})
                if src not in used_sources:
                    used_sources.append(src)

    if not bullets:
        return _low_confidence(status="weak_evidence")

    # short answer = first two bullets stitched
    answer = _make_short_answer([b["text"] for b in bullets[:2]])

    return {
        "answer": answer,
        "confidence": 0.4,
        "used_sources": used_sources[:2],
        "quotes": bullets[:MAX_QUOTES],
        "status": "ok",
        "debug": {"fallback_mode": "bullets", "picked_quotes": min(len(bullets), MAX_QUOTES)},
    }
