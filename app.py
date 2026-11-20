"""RAG chatbot over scraped DarGlobal + Wasalt data. BM25 retrieval + free OpenRouter model."""
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatbot")

BASE = Path(__file__).parent
OR_URL = "https://openrouter.ai/api/v1"
# Free models change often, so we discover them at runtime; these are only a last-resort fallback.
FALLBACK_MODELS = ["meta-llama/llama-3.3-70b-instruct:free", "google/gemma-3-27b-it:free",
                   "mistralai/mistral-small-3.2-24b-instruct:free", "openai/gpt-oss-20b:free"]
PREFERRED = ["llama-3.3", "gemma-3", "mistral-small", "qwen", "gpt-oss", "deepseek", "llama"]
BAD = ("guard", "embed", "moderation", "vision", "-vl", "image")
STOP = set("a an the of and or to in is are was were for on with at by from this that it as be "
           "what who how do does can you me about tell please".split())

SYSTEM = (
    "You are an assistant for DarGlobal (an international luxury real-estate developer) and "
    "Wasalt (a Saudi real-estate platform). Answer ONLY from the numbered context excerpts "
    "provided, and cite them like [1] or [2]. If the answer is not in the excerpts, say you "
    "could not find it in the collected data and suggest checking darglobal.co.uk or wasalt.sa. "
    "If the question is unrelated to DarGlobal or Wasalt, say you can only answer questions about them. "
    "State only what the excerpts say: do not infer facts (such as headquarters, ownership or partners) "
    "that are not written there. Be concise; never invent prices, dates or project details."
)


SYN = {"brand": "partners partnership", "brands": "partners partnership", "partner": "partners partnership",
       "owner": "joint venture owned founded", "owns": "joint venture owned founded",
       "owned": "joint venture owned founded", "ownership": "joint venture owned founded",
       "founder": "founder chief executive", "ceo": "chief executive officer founder",
       "where": "location located", "cost": "price starting", "price": "price starting from"}


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


# ---------- data loading / indexing ----------
def load_docs():
    docs = []
    for f in sorted((BASE / "data").glob("*.json")):
        docs += json.loads(f.read_text(encoding="utf-8"))
    by_src = {}
    for d in docs:
        by_src.setdefault(d["source"], []).append(d)
    for ds in by_src.values():  # drop long boilerplate lines repeated across many pages
        if len(ds) < 8:
            continue
        counts = Counter(l for d in ds for l in set(d["text"].split("\n")))
        common = {l for l, n in counts.items() if len(l) > 40 and n > 0.25 * len(ds)}
        for d in ds:
            d["text"] = "\n".join(l for l in d["text"].split("\n") if l not in common)
    return docs


def chunk(text, size=900):
    out, cur = [], ""
    for p in (p for p in text.split("\n") if p.strip()):
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = ""
        cur += p + "\n"
    if cur.strip():
        out.append(cur)
    return out


CHUNKS = []
for d in load_docs():
    for c in chunk(d["text"]):
        CHUNKS.append({"text": c, "title": d.get("title", ""), "url": d["url"], "source": d["source"]})
def slug(u):
    return re.sub(r"[^a-z0-9]+", " ", urlparse(u).path.lower())


BM25 = BM25Okapi([tok(f'{c["title"]} {slug(c["url"])} {c["title"]} {slug(c["url"])} {c["text"]}') for c in CHUNKS]) if CHUNKS else None
log.info("Indexed %d chunks", len(CHUNKS))


def retrieve(query, k=6):
    if not BM25:
        return []
    words = tok(query)
    words += [x for w in list(words) for x in SYN.get(w, "").split()]
    scores = BM25.get_scores(words)
    picked, per_url = [], Counter()
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if scores[i] <= 0 or len(picked) >= k:
            break
        if per_url[CHUNKS[i]["url"]] >= 2:  # keep sources diverse
            continue
        per_url[CHUNKS[i]["url"]] += 1
        picked.append(CHUNKS[i])
    return picked


# ---------- OpenRouter ----------
_cache = {"ids": [], "ts": 0.0}
_cool = {}  # model -> time until which we skip it
_last_good = {"id": None}


def candidate_models():
    if time.time() - _cache["ts"] > 3600:
        try:
            data = httpx.get(f"{OR_URL}/models", timeout=15).json()["data"]
            free = [m for m in data if m["id"].endswith(":free") and (m.get("context_length") or 0) >= 16000
                    and not any(b in m["id"] for b in BAD)]
            rank = lambda m: next((i for i, p in enumerate(PREFERRED) if p in m["id"]), len(PREFERRED))
            free.sort(key=lambda m: (rank(m), -(m.get("context_length") or 0)))
            _cache["ids"] = [m["id"] for m in free]
            _cache["ts"] = time.time()
        except Exception as e:
            log.warning("model list failed: %s", e)
    ids = ([os.getenv("OPENROUTER_MODEL")] if os.getenv("OPENROUTER_MODEL") else []) + _cache["ids"][:6] + FALLBACK_MODELS
    if _last_good["id"]:
        ids.insert(0, _last_good["id"])
    ids = [m for m in dict.fromkeys(ids) if _cool.get(m, 0) < time.time()]
    return ids[:5] or FALLBACK_MODELS[:3]


def call_llm(messages):
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": os.getenv("APP_URL", "https://huggingface.co"), "X-Title": "DarGlobal-Wasalt Chatbot"}
    last = None
    for model in candidate_models():
        try:
            r = httpx.post(f"{OR_URL}/chat/completions", headers=headers, timeout=45,
                           json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 700})
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if text:
                    _last_good["id"] = model
                    return text, model
            last = f"{model}: HTTP {r.status_code} {r.text[:150]}"
        except Exception as e:
            last = f"{model}: {e}"
        _cool[model] = time.time() + 300
        if _last_good["id"] == model:
            _last_good["id"] = None
        log.warning("model failed -> %s", last[:160])
    raise RuntimeError(last or "no model available")


# ---------- API ----------
class Turn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str
    history: list[Turn] = []


app = FastAPI(title="DarGlobal & Wasalt Chatbot")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS)}


@app.post("/chat")
def chat(body: ChatIn):
    q = body.message.strip()[:1000]
    prev_user = [t.content for t in body.history if t.role == "user"]
    query = (prev_user[-1] + " " + q) if prev_user and len(q.split()) < 6 else q  # help short follow-ups
    hits = retrieve(query)
    if not hits:
        return {"answer": "I couldn't find anything about that in the collected DarGlobal and Wasalt data.",
                "sources": [], "model": None}
    context = "\n\n".join(f"[{i}] ({h['source']}: {h['title']}) {h['url']}\n{h['text'][:1200]}"
                          for i, h in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM}]
    messages += [{"role": t.role, "content": t.content[:800]} for t in body.history[-6:] if t.role in ("user", "assistant")]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"})
    sources = [{"n": i, "title": h["title"] or h["url"], "url": h["url"], "source": h["source"]} for i, h in enumerate(hits, 1)]
    try:
        answer, model = call_llm(messages)
    except Exception as e:
        log.error("LLM failure: %s", e)
        top = hits[0]["text"][:500].strip()
        return {"answer": "The language model is busy right now (free models are rate-limited). "
                          f"Most relevant excerpt I found:\n\n{top}", "sources": sources[:1], "model": None}
    return {"answer": answer, "sources": sources, "model": model}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")