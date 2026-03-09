#!/usr/bin/env python3
"""
OCR Verification Test: 10 questions against live chatbot with source document verification.
"""
import asyncio
import json
import sys
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

QUESTIONS = [
    "한국가스공사법 제1조의 목적은 무엇인가요?",
    "한국가스공사의 사업 범위는 무엇인가요? 제11조를 인용해주세요.",
    "한국가스공사법 시행령 제1조의 목적은?",
    "감사규정의 목적은 무엇인가요?",
    "인사규정에서 정하는 징계의 종류는 무엇인가요?",
    "연봉의 구성요소는 무엇인가요?",
    "한국가스기술공사의 설립 연도와 주요 사업은?",
    "제32기말 유동자산은 얼마인가요?",
    "도시가스사업법 제1조의 목적은?",
    "고압가스안전관리법의 목적은 무엇인가요?",
]


async def query_chatbot(question: str, timeout_sec: int = 120) -> dict:
    """Send question to chatbot SSE endpoint and parse response."""
    result = {
        "answer": "",
        "sources": [],
        "confidence": None,
        "confidence_level": "",
        "latency_ms": 0,
        "error": None,
    }

    url = "http://localhost:8000/api/chat/stream"
    payload = {"message": question, "mode": "auto", "temperature": 0.2}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                buffer = ""
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if event_type == "chunk":
                            result["answer"] += data.get("content", "")
                        elif event_type == "source":
                            result["sources"].append({
                                "filename": data.get("filename", "?"),
                                "page": data.get("page"),
                                "score": data.get("score", 0),
                                "content_preview": data.get("content", "")[:300],
                            })
                        elif event_type == "end":
                            result["confidence"] = data.get("confidence_score")
                            result["confidence_level"] = data.get("confidence_level", "")
                            result["latency_ms"] = data.get("latency_ms", 0)
                        elif event_type == "error":
                            result["error"] = data.get("message", str(data))
    except Exception as e:
        result["error"] = str(e)

    return result


async def retrieve_sources(question: str, top_k: int = 3) -> list:
    """Retrieve source documents for verification."""
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever()
    results = await retriever.retrieve(query=question, top_k=top_k)
    sources = []
    for r in results:
        sources.append({
            "filename": r.metadata.get("filename", "?"),
            "page": r.metadata.get("page_number"),
            "score": round(r.score, 3),
            "content": r.content[:500],
        })
    return sources


async def run_all():
    results = []

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {q}")
        print(f"{'='*60}")

        # Query chatbot
        print(f"  [Querying chatbot...]")
        chat_result = await query_chatbot(q)

        if chat_result["error"]:
            print(f"  [ERROR]: {chat_result['error']}")

        # Print answer
        answer = chat_result["answer"].strip()
        print(f"\n  [Answer]: {answer[:500]}{'...' if len(answer) > 500 else ''}")
        conf = chat_result["confidence"]
        print(f"  [Confidence]: {conf*100:.1f}% ({chat_result['confidence_level']})" if conf else "  [Confidence]: N/A")
        print(f"  [Latency]: {chat_result['latency_ms']}ms")

        # Print sources from chatbot
        print(f"\n  [Chatbot Sources ({len(chat_result['sources'])})]:")
        for j, s in enumerate(chat_result["sources"][:5], 1):
            print(f"    {j}. {s['filename']} (score: {s['score']:.3f}, page: {s['page']})")

        # Print top source content for verification
        if chat_result["sources"]:
            top_src = chat_result["sources"][0]
            print(f"\n  [Top Source Content Preview]:")
            print(f"    {top_src['content_preview'][:300]}")

        results.append({
            "question": q,
            "answer": answer,
            "confidence": conf,
            "confidence_level": chat_result["confidence_level"],
            "latency_ms": chat_result["latency_ms"],
            "sources": chat_result["sources"],
            "error": chat_result["error"],
        })

    # Output JSON results for post-processing
    output_path = Path(__file__).parent.parent / "data" / "ocr_verification_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Q#':<4} {'Conf%':<8} {'Level':<8} {'Latency':<10} {'Sources':<8} {'Status'}")
    print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for i, r in enumerate(results, 1):
        conf_str = f"{r['confidence']*100:.1f}" if r['confidence'] else "N/A"
        status = "ERROR" if r['error'] else ("OK" if r['confidence'] and r['confidence'] > 0.5 else "LOW")
        print(f"Q{i:<3} {conf_str:<8} {r['confidence_level']:<8} {r['latency_ms']:<10} {len(r['sources']):<8} {status}")


if __name__ == "__main__":
    asyncio.run(run_all())
