import json
import re
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx

import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

HEADERS = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = (
    "You are a highly reliable Grounded QA API for medical and legal compliance.\n"
    "Answer the user's question strictly using ONLY the provided context chunks.\n"
    "Rules:\n"
    "1. If the question CANNOT be answered from the chunks, you MUST return:\n"
    "   - answerable: false\n"
    "   - answer: \"I don't know\" (exact text)\n"
    "   - citations: [] (empty array)\n"
    "   - confidence: a number <= 0.3\n"
    "2. If it CAN be answered using only the chunks, return:\n"
    "   - answerable: true\n"
    "   - answer: a concise, grounded answer using only chunk content\n"
    "   - citations: the chunk_id values you actually used (only real ids from the input)\n"
    "   - confidence: a number between 0.8 and 1.0\n"
    "3. NEVER use outside knowledge, even if you know the real-world answer.\n"
    "Return strictly valid JSON with exactly these 4 keys: answer, citations, confidence, answerable."
)


def parse_json_safely(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def fallback_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False,
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"ok": True, "email": config.EMAIL}


@app.post("/grounded-answer")
async def grounded_answer(request: Request):
    try:
        body = await request.json()
    except Exception:
        return fallback_response()

    question = (body or {}).get("question")
    chunks = (body or {}).get("chunks")

    if not question or not isinstance(chunks, list) or len(chunks) == 0:
        return fallback_response()

    valid_ids = {c.get("chunk_id") for c in chunks if isinstance(c, dict) and c.get("chunk_id")}

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT CHUNKS (JSON):\n{json.dumps(chunks, indent=2)}"
    )

    payload = {
        "model": config.TEXT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                f"{config.AIPIPE_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
            )
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"]
        parsed = parse_json_safely(raw_content)
        if parsed is None:
            return fallback_response()

        answerable = bool(parsed.get("answerable", False))
        confidence = parsed.get("confidence", 0.1)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.1

        if not answerable or confidence <= 0.3:
            return fallback_response()

        # Only allow citations that are real chunk IDs from the request.
        raw_citations = parsed.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
        citations = [c for c in raw_citations if c in valid_ids]

        answer_text = parsed.get("answer", "I don't know")
        if not isinstance(answer_text, str) or not answer_text.strip():
            return fallback_response()

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": max(0.0, min(1.0, confidence)),
            "answerable": True,
        }

    except (httpx.TimeoutException, httpx.HTTPError, KeyError, IndexError, asyncio.TimeoutError):
        return fallback_response()
    except Exception:
        return fallback_response()
