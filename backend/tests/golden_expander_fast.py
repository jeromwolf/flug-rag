"""골든 데이터셋 빠른 확장기 (템플릿 기반, LLM 불필요).

기존 골든 데이터를 시드로 질문 패러프레이징 + 변형을 수행한다.
LLM 호출 없이 수 초 내에 500문항 생성.

Usage:
    python tests/golden_expander_fast.py --target 500
"""

import argparse
import json
import random
import re
from pathlib import Path

DATASETS = [
    ("internal_rules", "tests/golden_dataset_internal_rules.json", 200),
    ("brochure", "tests/golden_dataset_brochure.json", 80),
    ("alio", "tests/golden_dataset_alio.json", 80),
    ("travel", "tests/golden_dataset_travel.json", 80),
]

# Question transformation templates
PREFIXES = [
    "",
    "한국가스기술공사의 ",
    "공사의 ",
    "KOGAS에서 ",
]

# Question reformulation patterns
REFORMULATIONS = [
    # (pattern, replacements)
    (r"^(.+)(?:은|는) 무엇(?:인가요|입니까|이에요)\?$", [
        "{0}에 대해 설명해 주세요.",
        "{0}의 정의는 무엇인가요?",
        "{0}을(를) 설명하시오.",
    ]),
    (r"^(.+)(?:은|는) 어떻게 (?:되나요|됩니까|하나요)\?$", [
        "{0}의 절차는 무엇인가요?",
        "{0}의 방법을 알려주세요.",
        "{0} 과정을 설명해 주세요.",
    ]),
    (r"^(.+)(?:에 대해|에 관해) (?:알려주세요|설명해 주세요|설명하시오)[.?]?$", [
        "{0}은(는) 무엇인가요?",
        "{0}에 대한 내용을 요약해 주세요.",
    ]),
    (r"^(.+)의 (?:종류|유형|분류)(?:는|은)?\s*(?:무엇|어떤 것)(?:인가요|입니까|이에요)\?$", [
        "{0}에는 어떤 유형이 있나요?",
        "{0}을(를) 분류해 주세요.",
    ]),
]

# Simple suffix transforms
SUFFIX_TRANSFORMS = [
    ("인가요?", ["입니까?", "인지 알려주세요.", "인지 설명해 주세요."]),
    ("입니까?", ["인가요?", "인지 설명해 주세요."]),
    ("하나요?", ["합니까?", "하는지 알려주세요."]),
    ("주세요.", ["주십시오.", "주세요?"]),
    ("무엇인가요?", ["무엇입니까?", "뭔가요?", "어떤 것인가요?"]),
]

# Difficulty variants
DIFFICULTY_PREFIXES = {
    "easy": ["", "간단히 ", "요약하면 "],
    "medium": ["", "구체적으로 "],
    "hard": ["상세히 ", "구체적으로 ", "근거와 함께 "],
}


def generate_variations(question: str, max_variants: int = 3) -> list[str]:
    """Generate question variations using template-based transforms."""
    variants = set()

    # 1. Suffix transforms
    for suffix, replacements in SUFFIX_TRANSFORMS:
        if question.endswith(suffix):
            for repl in replacements:
                variant = question[:-len(suffix)] + repl
                if variant != question:
                    variants.add(variant)

    # 2. Regex-based reformulations
    for pattern, templates in REFORMULATIONS:
        m = re.match(pattern, question)
        if m:
            groups = m.groups()
            for tmpl in templates:
                try:
                    variant = tmpl.format(*groups)
                    if variant != question:
                        variants.add(variant)
                except (IndexError, KeyError):
                    pass

    # 3. Prefix additions
    for prefix in PREFIXES:
        if prefix and not question.startswith(prefix):
            # Only add prefix if question starts with a topic word
            variant = prefix + question[0].lower() + question[1:]
            if variant != question and len(variant) < 200:
                variants.add(variant)

    # 4. "~에 대해" insertion
    if "에 대해" not in question and "에 관해" not in question:
        # Try inserting "에 대해" before the verb
        for verb in ["설명", "알려", "말씀"]:
            if verb in question:
                variant = question.replace(verb, "에 대해 " + verb, 1)
                if variant != question:
                    variants.add(variant)

    # 5. Negation variant for factual questions
    if "negative" not in question.lower():
        for keyword in ["있나요", "있습니까", "인가요", "맞나요"]:
            if keyword in question:
                neg_variant = question.replace(keyword, "없나요" if "있" in keyword else "아닌가요", 1)
                if neg_variant != question:
                    variants.add(neg_variant)
                break

    result = list(variants)[:max_variants]
    return result


def expand_dataset(seed_data: list, target_count: int, dataset_name: str) -> list:
    """Expand a single dataset to target count."""
    current = list(seed_data)
    needed = target_count - len(current)
    if needed <= 0:
        print(f"  [{dataset_name}] 이미 {len(current)}개 — 추가 불필요")
        return current

    print(f"  [{dataset_name}] {len(current)}개 → {target_count}개 (추가 {needed}개)")

    next_id = max((q.get("id", 0) for q in current), default=0) + 1
    generated = []

    # Round-robin through seeds, generating variations
    seed_idx = 0
    max_rounds = 10  # Safety limit
    round_num = 0

    while len(generated) < needed and round_num < max_rounds:
        round_num += 1
        for seed in current:
            if len(generated) >= needed:
                break

            variants = generate_variations(seed["question"], max_variants=3)
            if not variants:
                continue

            # Pick one variant (cycle through in successive rounds)
            variant_idx = (round_num - 1) % max(1, len(variants))
            if variant_idx >= len(variants):
                continue

            variant_q = variants[variant_idx]

            # Check uniqueness
            existing_questions = {q["question"] for q in current + generated}
            if variant_q in existing_questions:
                continue

            # Determine difficulty based on round
            difficulties = ["easy", "medium", "hard"]
            difficulty = difficulties[round_num % len(difficulties)]

            generated.append({
                "id": next_id,
                "category": seed.get("category", "factual"),
                "question": variant_q,
                "answer": seed["answer"],  # Same answer (paraphrase of same question)
                "difficulty": difficulty,
                "source_regulation": seed.get("source_regulation", ""),
                "source_document": seed.get("source_document", ""),
                "generated_from": seed.get("id", 0),
            })
            next_id += 1

    print(f"  [{dataset_name}] {len(generated)}개 추가 완료")
    return current + generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Total target question count")
    args = parser.parse_args()

    # Calculate per-dataset targets proportionally
    total_target = args.target
    total_existing = 0
    dataset_info = []

    for name, path_str, default_target in DATASETS:
        filepath = Path(__file__).parent.parent / path_str
        if not filepath.exists():
            print(f"[건너뜀] {path_str} 파일 없음")
            continue

        with open(filepath) as f:
            data = json.load(f)

        items = data.get("questions", data) if isinstance(data, dict) else data
        total_existing += len(items)
        dataset_info.append((name, filepath, items, default_target))

    # Scale targets proportionally to hit total
    if total_existing >= total_target:
        print(f"이미 {total_existing}문항 — 확장 불필요")
        return

    scale = total_target / sum(d[3] for d in dataset_info)

    total_generated = 0
    for name, filepath, items, default_target in dataset_info:
        target = max(len(items), int(default_target * scale))

        expanded = expand_dataset(items, target, name)
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
    main()
