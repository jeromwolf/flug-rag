"""골든 데이터셋 자동 생성기.

LLM을 활용하여 ChromaDB에 인제스트된 문서 청크로부터
RAG 평가용 골든 Q&A 데이터셋을 자동 생성한다.

카테고리: factual, inference, multi_hop, negative

Usage:
    python tests/golden_generator.py -n "내부규정" --filename-pattern ".*규정.*"
    python tests/golden_generator.py -n "출장보고서" -o data/golden_travel.json --counts '{"factual":8,"inference":6,"multi_hop":4,"negative":2}'
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import create_llm  # noqa: E402
from core.vectorstore import create_vectorstore  # noqa: E402
from rag.evaluator import AnswerEvaluator  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompt Constants (Korean)
# ---------------------------------------------------------------------------

FACTUAL_SYSTEM_PROMPT = """당신은 RAG 시스템 평가를 위한 골든 데이터셋 생성 전문가입니다.
주어진 텍스트에서 구체적 사실(숫자, 날짜, 목록, 정의, 명칭 등)을 추출하여
질문과 간결한 답변 쌍을 생성하세요.

[답변 스타일]
- 1~3문장으로 간결하게
- 원문의 표현을 최대한 그대로 사용
- 조문번호가 있으면 (제N조) 형식으로 포함
- 불필요한 서론이나 부연 설명 없이

[출력 형식]
JSON 형식으로만 응답하세요:
{"question": "질문 텍스트", "answer": "답변 텍스트"}"""

FACTUAL_USER_TEMPLATE = """다음 텍스트에서 구체적 사실(숫자, 날짜, 목록, 정의 등)을 찾아
질문과 1~3문장의 간결한 답변을 생성하세요.

[출처 문서: {source_name}]

---
{chunk_text}
---

JSON 형식으로만 응답하세요: {{"question": "...", "answer": "..."}}"""

INFERENCE_SYSTEM_PROMPT = """당신은 RAG 시스템 평가를 위한 골든 데이터셋 생성 전문가입니다.
주어진 텍스트를 바탕으로 '왜' 또는 '어떤 의미가 있는지' 묻는 추론 질문과
분석적 답변을 생성하세요.

[답변 스타일]
- 2~3문장으로 분석적으로
- 텍스트에 근거한 추론 (없는 사실 추가 금지)
- 원문 표현 인용 시 조문번호 포함
- 정책/절차의 목적이나 함의를 설명

[출력 형식]
JSON 형식으로만 응답하세요:
{"question": "질문 텍스트", "answer": "답변 텍스트"}"""

INFERENCE_USER_TEMPLATE = """다음 텍스트를 바탕으로 '왜' 또는 '어떤 의미가 있는지' 묻는
추론 질문과 2~3문장의 분석적 답변을 생성하세요.

[출처 문서: {source_name}]

---
{chunk_text}
---

JSON 형식으로만 응답하세요: {{"question": "...", "answer": "..."}}"""

MULTI_HOP_SYSTEM_PROMPT = """당신은 RAG 시스템 평가를 위한 골든 데이터셋 생성 전문가입니다.
여러 문서의 정보를 연결하여 답해야 하는 다중 홉(multi-hop) 질문을 생성하세요.

[답변 스타일]
- 2~4문장으로 종합적으로
- 각 문서에서 가져온 정보를 명확히 연결
- 출처 문서명이나 조문번호 포함
- 비교, 대조, 또는 종합 분석 형태

[출력 형식]
JSON 형식으로만 응답하세요:
{"question": "질문 텍스트", "answer": "답변 텍스트"}"""

MULTI_HOP_USER_TEMPLATE = """다음 여러 문서의 텍스트를 읽고, 2개 이상의 문서 정보를
연결해야 답할 수 있는 질문과 종합적 답변을 생성하세요.

{chunks_text}

JSON 형식으로만 응답하세요: {{"question": "...", "answer": "..."}}"""

NEGATIVE_SYSTEM_PROMPT = """당신은 RAG 시스템 평가를 위한 골든 데이터셋 생성 전문가입니다.
주어진 텍스트가 다루는 주제 영역에서, 텍스트에 포함되어 있지 않은
관련 주제에 대해 질문하세요.

[답변 스타일]
- "해당 문서에서 확인되지 않습니다" 또는 "규정에서 다루고 있지 않습니다"로 시작
- 1~2문장으로 간결하게
- 텍스트에 실제로 없는 내용에 대해서만 질문
- 비슷하지만 다른 주제로 질문 (예: 인사규정이 있으면 → 퇴직금 계산법)

[출력 형식]
JSON 형식으로만 응답하세요:
{"question": "질문 텍스트", "answer": "답변 텍스트"}"""

NEGATIVE_USER_TEMPLATE = """다음 텍스트가 다루는 주제 영역에서, 텍스트에 포함되어 있지 않은
관련 주제에 대해 질문하세요. 답변은 '확인되지 않습니다'로 시작하세요.

[출처 문서: {source_name}]

---
{chunk_text}
---

JSON 형식으로만 응답하세요: {{"question": "...", "answer": "..."}}"""

SELF_CONSISTENCY_SYSTEM_PROMPT = """당신은 RAG 시스템입니다. 주어진 문서를 참고하여
질문에 간결하게 답변하세요. 문서에 없는 내용은 답하지 마세요.

JSON 형식으로만 응답하세요:
{"answer": "답변 텍스트"}"""

SELF_CONSISTENCY_USER_TEMPLATE = """[참고 문서]
{context}

[질문]
{question}

JSON 형식으로만 응답하세요: {{"answer": "..."}}"""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """골든 데이터셋 생성 설정."""

    dataset_name: str
    dataset_version: str = "1.0"
    source_description: str = ""
    category_distribution: dict[str, int] = field(default_factory=lambda: {
        "factual": 10, "inference": 8, "multi_hop": 5, "negative": 3,
    })
    source_filter: dict | None = None
    filename_pattern: str | None = None
    max_chunks_per_document: int = 10
    min_chunk_length: int = 100
    temperature: float = 0.5
    max_retries: int = 2
    min_answer_length: int = 10
    max_answer_length: int = 500
    min_self_consistency: float = 0.6


@dataclass
class GeneratedQA:
    """생성된 Q&A 쌍."""

    question: str
    answer: str
    category: str
    difficulty: str
    source_document: str
    source_chunk_ids: list[str]
    generation_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GoldenDatasetGenerator:
    """LLM 기반 골든 데이터셋 자동 생성기.

    ChromaDB에서 문서 청크를 로드하고, LLM으로 카테고리별 Q&A 쌍을
    생성한 뒤, 자기일관성(self-consistency) 검증을 거쳐 최종 데이터셋을
    JSON으로 출력한다.
    """

    def __init__(
        self,
        config: GenerationConfig,
        llm=None,
        vectorstore=None,
        evaluator=None,
        skip_validation: bool = False,
    ):
        self.config = config
        self.llm = llm or create_llm(temperature=config.temperature)
        self.vectorstore = vectorstore or create_vectorstore()
        self.evaluator = evaluator or AnswerEvaluator()
        self.skip_validation = skip_validation
        self._generated_questions: list[str] = []  # dedup tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self) -> dict:
        """메인 생성 파이프라인. 완성된 골든 데이터셋 dict를 반환한다."""
        print(f"\n{'='*60}")
        print(f"골든 데이터셋 생성 시작: {self.config.dataset_name}")
        print(f"{'='*60}")

        # 1. 청크 로드
        doc_groups = await self._load_chunks()
        if not doc_groups:
            print("[오류] 필터 조건에 맞는 문서 청크가 없습니다.")
            return self._format_output([])

        total_chunks = sum(len(chunks) for chunks in doc_groups.values())
        print(f"[로드 완료] {len(doc_groups)}개 문서, {total_chunks}개 청크")

        # 2. 카테고리별 생성
        all_qa: list[GeneratedQA] = []

        # Flatten chunks for single-doc categories
        all_chunks: list[tuple[str, dict]] = []
        for source_name, chunks in doc_groups.items():
            for chunk in chunks:
                all_chunks.append((source_name, chunk))

        # factual
        count = self.config.category_distribution.get("factual", 0)
        if count > 0:
            print(f"\n--- factual ({count}개) 생성 ---")
            qa_list = await self._generate_factual(all_chunks, count)
            all_qa.extend(qa_list)
            print(f"[factual 완료] {len(qa_list)}/{count}개 생성")

        # inference
        count = self.config.category_distribution.get("inference", 0)
        if count > 0:
            print(f"\n--- inference ({count}개) 생성 ---")
            qa_list = await self._generate_inference(all_chunks, count)
            all_qa.extend(qa_list)
            print(f"[inference 완료] {len(qa_list)}/{count}개 생성")

        # multi_hop
        count = self.config.category_distribution.get("multi_hop", 0)
        if count > 0:
            print(f"\n--- multi_hop ({count}개) 생성 ---")
            qa_list = await self._generate_multi_hop(doc_groups, count)
            all_qa.extend(qa_list)
            print(f"[multi_hop 완료] {len(qa_list)}/{count}개 생성")

        # negative
        count = self.config.category_distribution.get("negative", 0)
        if count > 0:
            print(f"\n--- negative ({count}개) 생성 ---")
            qa_list = await self._generate_negative(all_chunks, count)
            all_qa.extend(qa_list)
            print(f"[negative 완료] {len(qa_list)}/{count}개 생성")

        # 3. 최종 요약
        print(f"\n{'='*60}")
        print(f"생성 완료: 총 {len(all_qa)}개 Q&A")
        for cat in ("factual", "inference", "multi_hop", "negative"):
            cat_count = sum(1 for q in all_qa if q.category == cat)
            if cat_count > 0:
                print(f"  {cat}: {cat_count}개")
        print(f"{'='*60}\n")

        return self._format_output(all_qa)

    # ------------------------------------------------------------------
    # Chunk Loading
    # ------------------------------------------------------------------

    async def _load_chunks(self) -> dict[str, list[dict]]:
        """청크를 로드하고 설정에 따라 필터링, 문서별로 그룹화한다."""
        all_docs = await self.vectorstore.get_all_documents()
        print(f"[벡터스토어] 전체 {len(all_docs)}개 청크 로드")

        # Filter by source_filter (metadata match)
        if self.config.source_filter:
            filtered = []
            for doc in all_docs:
                meta = doc.get("metadata", {})
                match = all(
                    meta.get(k) == v
                    for k, v in self.config.source_filter.items()
                )
                if match:
                    filtered.append(doc)
            all_docs = filtered
            print(f"[메타데이터 필터] {len(all_docs)}개 청크 남음")

        # Filter by filename_pattern
        if self.config.filename_pattern:
            pattern = re.compile(self.config.filename_pattern)
            filtered = []
            for doc in all_docs:
                meta = doc.get("metadata", {})
                filename = meta.get("filename", meta.get("source", ""))
                if pattern.search(filename):
                    filtered.append(doc)
            all_docs = filtered
            print(f"[파일명 필터 '{self.config.filename_pattern}'] {len(all_docs)}개 청크 남음")

        # Filter by min chunk length
        all_docs = [
            doc for doc in all_docs
            if len(doc.get("content", "")) >= self.config.min_chunk_length
        ]

        # Group by document filename
        doc_groups: dict[str, list[dict]] = {}
        for doc in all_docs:
            meta = doc.get("metadata", {})
            filename = meta.get("filename", meta.get("source", "unknown"))
            doc_groups.setdefault(filename, []).append(doc)

        # Limit chunks per document
        max_per_doc = self.config.max_chunks_per_document
        for filename in doc_groups:
            chunks = doc_groups[filename]
            if len(chunks) > max_per_doc:
                doc_groups[filename] = random.sample(chunks, max_per_doc)

        return doc_groups

    # ------------------------------------------------------------------
    # Factual Generation
    # ------------------------------------------------------------------

    async def _generate_factual(
        self,
        all_chunks: list[tuple[str, dict]],
        count: int,
    ) -> list[GeneratedQA]:
        """사실 기반 Q&A 생성. 숫자, 날짜, 목록, 정의가 포함된 청크를 우선 선택."""
        # Prefer chunks with factual signals
        scored = []
        for source_name, chunk in all_chunks:
            content = chunk.get("content", "")
            score = self._factual_signal_score(content)
            scored.append((score, source_name, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select top candidates, then sample for diversity
        candidates = scored[:max(count * 5, len(scored))]
        if len(candidates) > count * 3:
            candidates = random.sample(candidates, count * 3)

        results: list[GeneratedQA] = []
        attempt = 0
        for score, source_name, chunk in candidates:
            if len(results) >= count:
                break
            attempt += 1
            print(f"  [factual {len(results)+1}/{count}] 생성 중... (문서: {source_name[:30]})")

            qa = await self._generate_single_qa(
                system_prompt=FACTUAL_SYSTEM_PROMPT,
                user_prompt=FACTUAL_USER_TEMPLATE.format(
                    source_name=source_name,
                    chunk_text=chunk["content"][:2000],
                ),
                category="factual",
                source_name=source_name,
                source_chunks=[chunk],
            )
            if qa is not None:
                results.append(qa)

        return results

    # ------------------------------------------------------------------
    # Inference Generation
    # ------------------------------------------------------------------

    async def _generate_inference(
        self,
        all_chunks: list[tuple[str, dict]],
        count: int,
    ) -> list[GeneratedQA]:
        """추론 Q&A 생성. 정책/절차/규정 텍스트를 우선 선택."""
        scored = []
        for source_name, chunk in all_chunks:
            content = chunk.get("content", "")
            score = self._inference_signal_score(content)
            scored.append((score, source_name, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = scored[:max(count * 5, len(scored))]
        if len(candidates) > count * 3:
            candidates = random.sample(candidates, count * 3)

        results: list[GeneratedQA] = []
        for score, source_name, chunk in candidates:
            if len(results) >= count:
                break
            print(f"  [inference {len(results)+1}/{count}] 생성 중... (문서: {source_name[:30]})")

            qa = await self._generate_single_qa(
                system_prompt=INFERENCE_SYSTEM_PROMPT,
                user_prompt=INFERENCE_USER_TEMPLATE.format(
                    source_name=source_name,
                    chunk_text=chunk["content"][:2000],
                ),
                category="inference",
                source_name=source_name,
                source_chunks=[chunk],
            )
            if qa is not None:
                results.append(qa)

        return results

    # ------------------------------------------------------------------
    # Multi-hop Generation
    # ------------------------------------------------------------------

    async def _generate_multi_hop(
        self,
        doc_groups: dict[str, list[dict]],
        count: int,
    ) -> list[GeneratedQA]:
        """다중 홉 Q&A 생성. 서로 다른 문서에서 2~3개 청크를 연결."""
        doc_names = list(doc_groups.keys())
        if len(doc_names) < 2:
            print("  [경고] 다중 홉 생성에 최소 2개 문서가 필요합니다.")
            # Fall back to chunks within same doc
            if doc_names:
                return await self._generate_multi_hop_single_doc(
                    doc_groups[doc_names[0]], doc_names[0], count,
                )
            return []

        results: list[GeneratedQA] = []
        max_attempts = count * 4

        for attempt_idx in range(max_attempts):
            if len(results) >= count:
                break

            # Pick 2-3 different documents
            n_docs = min(random.choice([2, 2, 3]), len(doc_names))
            selected_docs = random.sample(doc_names, n_docs)

            # Pick one chunk from each
            selected_chunks: list[tuple[str, dict]] = []
            for doc_name in selected_docs:
                chunk = random.choice(doc_groups[doc_name])
                selected_chunks.append((doc_name, chunk))

            # Build combined text
            chunks_text_parts = []
            for i, (doc_name, chunk) in enumerate(selected_chunks, 1):
                chunks_text_parts.append(
                    f"[문서 {i}: {doc_name}]\n{chunk['content'][:1500]}"
                )
            chunks_text = "\n\n".join(chunks_text_parts)

            print(f"  [multi_hop {len(results)+1}/{count}] 생성 중... "
                  f"(문서: {', '.join(d[:20] for d, _ in selected_chunks)})")

            qa = await self._generate_single_qa(
                system_prompt=MULTI_HOP_SYSTEM_PROMPT,
                user_prompt=MULTI_HOP_USER_TEMPLATE.format(chunks_text=chunks_text),
                category="multi_hop",
                source_name=" + ".join(d for d, _ in selected_chunks),
                source_chunks=[c for _, c in selected_chunks],
            )
            if qa is not None:
                results.append(qa)

        return results

    async def _generate_multi_hop_single_doc(
        self,
        chunks: list[dict],
        doc_name: str,
        count: int,
    ) -> list[GeneratedQA]:
        """단일 문서 내 다중 청크 연결 (문서가 1개뿐일 때 폴백)."""
        results: list[GeneratedQA] = []
        if len(chunks) < 2:
            return results

        for _ in range(count * 3):
            if len(results) >= count:
                break

            n = min(random.choice([2, 3]), len(chunks))
            selected = random.sample(chunks, n)

            chunks_text_parts = []
            for i, chunk in enumerate(selected, 1):
                chunks_text_parts.append(
                    f"[섹션 {i}]\n{chunk['content'][:1500]}"
                )
            chunks_text = "\n\n".join(chunks_text_parts)

            print(f"  [multi_hop {len(results)+1}/{count}] 생성 중... (문서: {doc_name[:30]})")

            qa = await self._generate_single_qa(
                system_prompt=MULTI_HOP_SYSTEM_PROMPT,
                user_prompt=MULTI_HOP_USER_TEMPLATE.format(chunks_text=chunks_text),
                category="multi_hop",
                source_name=doc_name,
                source_chunks=selected,
            )
            if qa is not None:
                results.append(qa)

        return results

    # ------------------------------------------------------------------
    # Negative Generation
    # ------------------------------------------------------------------

    async def _generate_negative(
        self,
        all_chunks: list[tuple[str, dict]],
        count: int,
    ) -> list[GeneratedQA]:
        """부정 Q&A 생성. 문서에 없는 관련 주제에 대해 질문."""
        # Prefer chunks with clear topic boundaries
        candidates = list(all_chunks)
        random.shuffle(candidates)
        if len(candidates) > count * 5:
            candidates = candidates[:count * 5]

        results: list[GeneratedQA] = []
        for source_name, chunk in candidates:
            if len(results) >= count:
                break
            print(f"  [negative {len(results)+1}/{count}] 생성 중... (문서: {source_name[:30]})")

            qa = await self._generate_single_qa(
                system_prompt=NEGATIVE_SYSTEM_PROMPT,
                user_prompt=NEGATIVE_USER_TEMPLATE.format(
                    source_name=source_name,
                    chunk_text=chunk["content"][:2000],
                ),
                category="negative",
                source_name=source_name,
                source_chunks=[chunk],
            )
            if qa is not None:
                results.append(qa)

        return results

    # ------------------------------------------------------------------
    # Core Generation + Validation
    # ------------------------------------------------------------------

    async def _generate_single_qa(
        self,
        system_prompt: str,
        user_prompt: str,
        category: str,
        source_name: str,
        source_chunks: list[dict],
    ) -> GeneratedQA | None:
        """단일 Q&A 쌍 생성 + 검증. 실패 시 재시도."""
        for retry in range(self.config.max_retries + 1):
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=self.config.temperature,
                )
                parsed = self._parse_json_response(response.content)
                if parsed is None:
                    logger.debug("JSON 파싱 실패 (시도 %d): %s", retry + 1, response.content[:200])
                    continue

                question = parsed.get("question", "").strip()
                answer = parsed.get("answer", "").strip()

                if not question or not answer:
                    continue

                # Basic length check
                if len(answer) < self.config.min_answer_length:
                    continue
                if len(answer) > self.config.max_answer_length:
                    # Truncate if slightly over, skip if way over
                    if len(answer) > self.config.max_answer_length * 1.5:
                        continue
                    answer = answer[:self.config.max_answer_length]

                # Dedup check
                if self._is_duplicate(question):
                    continue

                qa = GeneratedQA(
                    question=question,
                    answer=answer,
                    category=category,
                    difficulty="",  # assigned later
                    source_document=source_name,
                    source_chunk_ids=[c.get("id", "") for c in source_chunks],
                    generation_metadata={
                        "retry": retry,
                        "model": getattr(self.llm, "model", "unknown"),
                    },
                )

                # Validate
                if not self.skip_validation:
                    valid = await self._validate_qa_pair(qa, source_chunks)
                    if not valid:
                        continue

                # Assign difficulty
                qa.difficulty = self._assign_difficulty(qa, source_chunks)

                # Track for dedup
                self._generated_questions.append(question)
                return qa

            except Exception as e:
                logger.warning("Q&A 생성 오류 (시도 %d): %s", retry + 1, e)
                continue

        return None

    def _parse_json_response(self, content: str) -> dict | None:
        """LLM 응답에서 JSON을 추출한다."""
        content = content.strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block from markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... }
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    async def _validate_qa_pair(
        self,
        qa: GeneratedQA,
        source_chunks: list[dict],
    ) -> bool:
        """Q&A 쌍의 품질을 검증한다.

        1. 답변 길이 범위 확인
        2. 기존 질문과의 중복 확인 (Jaccard > 0.8)
        3. 자기일관성: 소스 청크를 컨텍스트로 LLM에 재질문, evaluator로 비교
        """
        # Length bounds (already checked in _generate_single_qa, double-check)
        if len(qa.answer) < self.config.min_answer_length:
            return False
        if len(qa.answer) > self.config.max_answer_length * 1.5:
            return False

        # Self-consistency check
        context = "\n\n".join(
            c.get("content", "")[:1500] for c in source_chunks
        )

        try:
            response = await self.llm.generate(
                prompt=SELF_CONSISTENCY_USER_TEMPLATE.format(
                    context=context,
                    question=qa.question,
                ),
                system=SELF_CONSISTENCY_SYSTEM_PROMPT,
                temperature=0.1,  # low temp for consistency
            )
            parsed = self._parse_json_response(response.content)
            if parsed is None:
                # If we can't parse the consistency check, use raw content
                regenerated_answer = response.content.strip()
            else:
                regenerated_answer = parsed.get("answer", response.content).strip()

            if not regenerated_answer:
                return False

            # Compare original answer with regenerated answer
            eval_result = await self.evaluator.evaluate(
                expected=qa.answer,
                actual=regenerated_answer,
            )
            consistency_score = eval_result.composite_score
            qa.generation_metadata["self_consistency"] = round(consistency_score, 4)

            if consistency_score < self.config.min_self_consistency:
                logger.debug(
                    "자기일관성 미달 (%.3f < %.3f): %s",
                    consistency_score,
                    self.config.min_self_consistency,
                    qa.question[:50],
                )
                return False

        except Exception as e:
            logger.warning("자기일관성 검증 오류: %s", e)
            # On validation error, accept the QA pair rather than losing it
            qa.generation_metadata["self_consistency"] = -1.0

        return True

    def _is_duplicate(self, question: str) -> bool:
        """기존 생성된 질문과의 중복 여부를 Jaccard 유사도로 확인."""
        new_tokens = set(question.replace(" ", ""))
        for existing_q in self._generated_questions:
            existing_tokens = set(existing_q.replace(" ", ""))
            if not new_tokens or not existing_tokens:
                continue
            intersection = new_tokens & existing_tokens
            union = new_tokens | existing_tokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard > 0.8:
                return True
        return False

    # ------------------------------------------------------------------
    # Difficulty Assignment
    # ------------------------------------------------------------------

    def _assign_difficulty(self, qa: GeneratedQA, source_chunks: list[dict]) -> str:
        """휴리스틱 기반 난이도 할당.

        easy:   단일 청크, 직접 추출, 짧은 답변
        medium: 문맥 이해 필요, 중간 길이
        hard:   multi_hop, 복잡 추론, 또는 negative
        """
        # negative and multi_hop are inherently harder
        if qa.category == "negative":
            return "medium"
        if qa.category == "multi_hop":
            return "hard" if len(source_chunks) >= 3 else "medium"

        answer_len = len(qa.answer)
        num_chunks = len(source_chunks)

        # Short answer from single chunk = easy
        if num_chunks <= 1 and answer_len <= 80:
            return "easy"

        # Long answer or multi-chunk = medium/hard
        if answer_len > 200 or num_chunks >= 3:
            return "hard"

        # Check for reasoning indicators
        reasoning_markers = ["따라서", "때문에", "의미는", "결과적으로", "즉,"]
        has_reasoning = any(m in qa.answer for m in reasoning_markers)
        if has_reasoning or qa.category == "inference":
            return "medium"

        return "easy"

    # ------------------------------------------------------------------
    # Chunk Scoring Heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _factual_signal_score(text: str) -> float:
        """청크에 사실 정보(숫자, 날짜, 목록 등)가 얼마나 많은지 점수화."""
        score = 0.0
        # Numbers (년, 월, 일, 원, %, 조, 항)
        score += len(re.findall(r"\d+[년월일원%]", text)) * 2.0
        # Article references (제N조, 제N항)
        score += len(re.findall(r"제\d+[조항호]", text)) * 3.0
        # Lists (1., 2., 가., 나., ①, ②)
        score += len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]\s|[가-힣][.)]\s|[①-⑳])", text)) * 1.5
        # Definitions (이라 함은, 말한다, 의미한다)
        score += len(re.findall(r"(?:이라\s*함은|말한다|의미한다|뜻한다)", text)) * 4.0
        # Proper nouns / organization names often in quotes
        score += len(re.findall(r"[「『""].*?[」』""]", text)) * 1.0
        return score

    @staticmethod
    def _inference_signal_score(text: str) -> float:
        """청크에 정책/절차/근거 텍스트가 얼마나 많은지 점수화."""
        score = 0.0
        # Policy/procedure keywords
        policy_words = [
            "하여야 한다", "할 수 있다", "하지 못한다",
            "목적", "원칙", "기준", "절차", "방법",
            "승인", "결재", "보고", "심의", "의결",
        ]
        for word in policy_words:
            score += text.count(word) * 2.0
        # Conditional logic
        score += len(re.findall(r"(?:경우에는|때에는|경우\s)", text)) * 2.5
        # Purpose/rationale markers
        score += len(re.findall(r"(?:위하여|위해|~함으로써|기여)", text)) * 3.0
        return score

    # ------------------------------------------------------------------
    # Output Formatting
    # ------------------------------------------------------------------

    def _format_output(self, questions: list[GeneratedQA]) -> dict:
        """골든 데이터셋 JSON 스키마로 포맷한다."""
        # Count per category
        category_counts: dict[str, int] = {}
        for qa in questions:
            category_counts[qa.category] = category_counts.get(qa.category, 0) + 1

        formatted_questions = []
        for idx, qa in enumerate(questions, 1):
            entry: dict = {
                "id": idx,
                "category": qa.category,
                "question": qa.question,
                "answer": qa.answer,
                "source_document": qa.source_document,
                "difficulty": qa.difficulty,
            }
            # Include generation metadata if present
            if qa.generation_metadata.get("self_consistency", -1) >= 0:
                entry["self_consistency"] = qa.generation_metadata["self_consistency"]
            formatted_questions.append(entry)

        return {
            "dataset_info": {
                "name": self.config.dataset_name,
                "version": self.config.dataset_version,
                "source": self.config.source_description,
                "total_questions": len(questions),
                "categories": category_counts,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "generator": "golden_generator.py",
                "generation_config": {
                    "temperature": self.config.temperature,
                    "min_self_consistency": self.config.min_self_consistency,
                    "model": getattr(self.llm, "model", "unknown"),
                    "skip_validation": self.skip_validation,
                },
            },
            "questions": formatted_questions,
        }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="골든 데이터셋 자동 생성기 - LLM 기반 Q&A 데이터셋 생성",
    )
    parser.add_argument(
        "--dataset-name", "-n",
        required=True,
        help="데이터셋 이름 (예: '한국가스기술공사 내부규정')",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 JSON 파일 경로 (기본값: tests/golden_dataset_{name}.json)",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="소스 필터 메타데이터 (JSON 문자열, 예: '{\"source_type\": \"regulation\"}')",
    )
    parser.add_argument(
        "--filename-pattern",
        type=str,
        default=None,
        help="파일명 정규식 필터 (예: '.*규정.*')",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help='카테고리별 수량 (JSON, 예: \'{"factual":10,"inference":8,"multi_hop":5,"negative":3}\')',
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM 모델 (예: 'qwen2.5:14b')",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.5,
        help="LLM 생성 온도 (기본값: 0.5)",
    )
    parser.add_argument(
        "--source-description",
        type=str,
        default="",
        help="데이터셋 출처 설명",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.6,
        help="최소 자기일관성 점수 (기본값: 0.6)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="품질 검증 건너뛰기 (빠른 생성용)",
    )

    args = parser.parse_args()

    # Parse category counts
    category_distribution = {"factual": 10, "inference": 8, "multi_hop": 5, "negative": 3}
    if args.counts:
        try:
            category_distribution = json.loads(args.counts)
        except json.JSONDecodeError:
            print(f"[오류] --counts JSON 파싱 실패: {args.counts}")
            sys.exit(1)

    # Parse source filter
    source_filter = None
    if args.source_filter:
        try:
            source_filter = json.loads(args.source_filter)
        except json.JSONDecodeError:
            print(f"[오류] --source-filter JSON 파싱 실패: {args.source_filter}")
            sys.exit(1)

    # Build config
    config = GenerationConfig(
        dataset_name=args.dataset_name,
        source_description=args.source_description,
        category_distribution=category_distribution,
        source_filter=source_filter,
        filename_pattern=args.filename_pattern,
        temperature=args.temperature,
        min_self_consistency=args.min_consistency,
    )

    # Create LLM
    llm = None
    if args.model:
        llm = create_llm(model=args.model, temperature=args.temperature)

    # Determine output path
    output_path = args.output
    if not output_path:
        safe_name = re.sub(r"[^\w가-힣]", "_", args.dataset_name).strip("_")
        output_path = str(
            Path(__file__).parent / f"golden_dataset_{safe_name}.json"
        )

    # Run generator
    async def run():
        generator = GoldenDatasetGenerator(
            config=config,
            llm=llm,
            skip_validation=args.skip_validation,
        )
        dataset = await generator.generate()

        # Write output
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        total = dataset["dataset_info"]["total_questions"]
        print(f"\n[저장 완료] {output}")
        print(f"  총 {total}개 Q&A 생성")

        if total == 0:
            print("  [경고] 생성된 Q&A가 없습니다. 필터 조건을 확인하세요.")

    asyncio.run(run())


if __name__ == "__main__":
    main()
