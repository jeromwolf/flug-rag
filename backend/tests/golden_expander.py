"""골든 데이터셋 빠른 확장기.

기존 골든 데이터를 시드로 LLM에게 변형 문항을 생성하게 한다.
Milvus 접속 불필요 — 기존 Q&A 기반 변형만 수행.

Usage:
    python tests/golden_expander.py --target 500
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import create_llm

DATASETS = [
    ("internal_rules", "tests/golden_dataset_internal_rules.json", 200),
    ("brochure", "tests/golden_dataset_brochure.json", 80),
    ("alio", "tests/golden_dataset_alio.json", 80),
    ("travel", "tests/golden_dataset_travel.json", 80),
]

EXPAND_SYSTEM = """당신은 RAG 시스템 평가용 골든 데이터셋 확장 전문가입니다.
주어진 Q&A 쌍을 바탕으로 유사하지만 다른 관점의 질문을 생성하세요.

[규칙]
- 같은 출처 문서/주제 범위 내에서 변형
- 카테고리(factual/inference/multi_hop/negative)를 유지
- 답변은 1~3문장으로 간결하게
- 원본과 겹치지 않는 새로운 질문
- 난이도를 다양화 (easy/medium/hard)

[출력]
JSON 배열로만 응답: [{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}]"""

EXPAND_USER = """다음 Q&A를 바탕으로 같은 주제에서 {count}개의 새로운 질문-답변 쌍을 생성하세요.

[카테고리: {category}]
[출처: {source}]

원본 Q: {question}
원본 A: {answer}

새로운 {count}개의 Q&A를 JSON 배열로 응답하세요."""


async def expand_dataset(llm, seed_data: list, target_count: int, dataset_name: str) -> list:
    """Expand a single dataset to target count."""
    current = list(seed_data)
    needed = target_count - len(current)
    if needed <= 0:
        print(f"  [{dataset_name}] 이미 {len(current)}개 — 추가 불필요")
        return current

    print(f"  [{dataset_name}] {len(current)}개 → {target_count}개 (추가 {needed}개)")

    next_id = max((q.get("id", 0) for q in current), default=0) + 1
    generated = []

    # Cycle through seed questions
    seed_idx = 0
    while len(generated) < needed:
        seed = current[seed_idx % len(current)]
        seed_idx += 1

        # Generate 2-3 variations per seed
        batch_count = min(3, needed - len(generated))

        prompt = EXPAND_USER.format(
            count=batch_count,
            category=seed.get("category", "factual"),
            source=seed.get("source_regulation", seed.get("source_document", "")),
            question=seed["question"],
            answer=seed["answer"],
        )

        try:
            resp = await llm.generate(prompt=prompt, system=EXPAND_SYSTEM, temperature=0.3, max_tokens=1024)
            text = resp.content if hasattr(resp, "content") else str(resp)

            # Parse JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            items = json.loads(text)
            if not isinstance(items, list):
                items = [items]

            for item in items:
                if not item.get("question") or not item.get("answer"):
                    continue
                generated.append({
                    "id": next_id,
                    "category": seed.get("category", "factual"),
                    "question": item["question"],
                    "answer": item["answer"],
                    "difficulty": item.get("difficulty", "medium"),
                    "source_regulation": seed.get("source_regulation", ""),
                    "source_document": seed.get("source_document", ""),
                    "generated_from": seed.get("id", 0),
                })
                next_id += 1

                if len(generated) >= needed:
                    break

        except Exception as e:
            print(f"    [경고] 생성 실패: {e}")
            continue

        if seed_idx % 10 == 0:
            print(f"    진행: {len(generated)}/{needed}")

    print(f"  [{dataset_name}] {len(generated)}개 추가 완료")
    return current + generated


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Total target question count")
    args = parser.parse_args()

    llm = create_llm(temperature=0.3)

    total_existing = 0
    total_generated = 0

    for name, path, target in DATASETS:
        filepath = Path(__file__).parent.parent / path
        if not filepath.exists():
            print(f"[건너뜀] {path} 파일 없음")
            continue

        with open(filepath) as f:
            data = json.load(f)

        items = data.get("questions", data) if isinstance(data, dict) else data
        total_existing += len(items)

        expanded = await expand_dataset(llm, items, target, name)
        total_generated += len(expanded) - len(items)

        # Save expanded dataset
        output_path = filepath.parent / f"golden_dataset_{name}_expanded.json"
        output_data = {
            "dataset_name": name,
            "total_count": len(expanded),
            "original_count": len(items),
            "generated_count": len(expanded) - len(items),
            "questions": expanded,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  저장: {output_path}")

    print(f"\n{'='*50}")
    print(f"완료: 기존 {total_existing}문항 + 신규 {total_generated}문항 = {total_existing + total_generated}문항")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
