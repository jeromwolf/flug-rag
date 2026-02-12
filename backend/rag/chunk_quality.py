"""Chunk quality analysis for RAG pipeline quality management (SFR-008)."""

import asyncio
import hashlib
import re
import statistics
from dataclasses import dataclass
from datetime import datetime

from core.vectorstore import create_vectorstore
from core.vectorstore.base import BaseVectorStore


@dataclass
class ChunkQualityReport:
    """종합 청크 품질 분석 결과."""
    total_chunks: int
    length_distribution: dict
    special_char_ratio: float
    duplicate_count: int
    near_duplicate_count: int
    empty_chunk_count: int
    table_chunk_count: int
    avg_semantic_completeness: float
    analyzed_at: str  # ISO datetime


@dataclass
class ChunkPreview:
    """청크 미리보기 정보."""
    id: str
    content_preview: str
    index: int
    length: int
    has_table: bool
    page_number: int | None


@dataclass
class DocumentChunkStats:
    """문서별 청크 통계."""
    document_id: str
    filename: str
    chunk_count: int
    avg_length: float
    min_length: int
    max_length: int


class ChunkQualityAnalyzer:
    """RAG 파이프라인 청크 품질 분석기."""

    def __init__(self, vectorstore: BaseVectorStore | None = None):
        """Initialize analyzer with optional vectorstore.

        Args:
            vectorstore: Vector store instance (defaults to create_vectorstore())
        """
        self.vectorstore = vectorstore or create_vectorstore()

    async def analyze_all_chunks(self, limit: int = 5000) -> ChunkQualityReport:
        """전체 청크 품질 분석.

        Args:
            limit: Maximum number of chunks to analyze (default 5000)

        Returns:
            ChunkQualityReport with comprehensive quality metrics
        """
        if not hasattr(self.vectorstore, '_collection'):
            raise RuntimeError("ChunkQualityAnalyzer requires ChromaDB vectorstore")

        # ChromaDB 컬렉션에서 모든 청크 가져오기 (문서와 메타데이터 포함)
        results = await asyncio.to_thread(
            self.vectorstore._collection.get,
            include=["documents", "metadatas"],
            limit=limit,
        )

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append({
                    "id": chunk_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        total_chunks = len(chunks)
        if total_chunks == 0:
            # 청크가 없는 경우 빈 리포트 반환
            return ChunkQualityReport(
                total_chunks=0,
                length_distribution={
                    "min": 0, "max": 0, "avg": 0, "median": 0, "std": 0,
                    "histogram": {"0-200": 0, "200-400": 0, "400-600": 0, "600-800": 0, "800+": 0},
                },
                special_char_ratio=0.0,
                duplicate_count=0,
                near_duplicate_count=0,
                empty_chunk_count=0,
                table_chunk_count=0,
                avg_semantic_completeness=0.0,
                analyzed_at=datetime.now().isoformat(),
            )

        # 청크 길이 수집
        lengths = [len(chunk["content"]) for chunk in chunks]

        # 길이 분포 계산
        length_dist = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            "histogram": self._build_histogram(lengths),
        }

        # 특수 문자 비율 계산 (비알파벳, 비한글, 비숫자 문자)
        special_char_ratio = self._calculate_special_char_ratio(chunks)

        # 중복 감지 (완전 중복)
        duplicate_count = self._count_duplicates(chunks)

        # 유사 중복 감지 (Jaccard similarity > 0.9)
        near_duplicate_count = self._count_near_duplicates(chunks)

        # 빈 청크 카운트 (길이 < 10)
        empty_chunk_count = sum(1 for chunk in chunks if len(chunk["content"].strip()) < 10)

        # 테이블 청크 카운트
        table_chunk_count = sum(
            1 for chunk in chunks
            if chunk["metadata"].get("has_table", False)
        )

        # 의미 완결성 (문장 종결 부호로 끝나는 청크 비율)
        avg_semantic_completeness = self._calculate_semantic_completeness(chunks)

        return ChunkQualityReport(
            total_chunks=total_chunks,
            length_distribution=length_dist,
            special_char_ratio=special_char_ratio,
            duplicate_count=duplicate_count,
            near_duplicate_count=near_duplicate_count,
            empty_chunk_count=empty_chunk_count,
            table_chunk_count=table_chunk_count,
            avg_semantic_completeness=avg_semantic_completeness,
            analyzed_at=datetime.now().isoformat(),
        )

    async def get_chunk_preview(self, document_id: str) -> list[ChunkPreview]:
        """특정 문서의 청크 미리보기.

        Args:
            document_id: Document ID to filter chunks

        Returns:
            List of ChunkPreview objects
        """
        if not hasattr(self.vectorstore, '_collection'):
            raise RuntimeError("ChunkQualityAnalyzer requires ChromaDB vectorstore")

        # ChromaDB where 필터로 해당 문서 청크만 가져오기
        results = await asyncio.to_thread(
            self.vectorstore._collection.get,
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

        previews = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""

                previews.append(
                    ChunkPreview(
                        id=chunk_id,
                        content_preview=content[:200],  # 처음 200자만
                        index=metadata.get("chunk_index", 0),
                        length=len(content),
                        has_table=metadata.get("has_table", False),
                        page_number=metadata.get("page_number"),
                    )
                )

        # chunk_index로 정렬
        previews.sort(key=lambda x: x.index)
        return previews

    async def get_chunks_by_document(self, limit: int = 10000) -> dict[str, DocumentChunkStats]:
        """문서별 청크 통계 집계.

        Args:
            limit: Maximum number of chunks to analyze (default 10000)

        Returns:
            Dict mapping document_id to DocumentChunkStats
        """
        if not hasattr(self.vectorstore, '_collection'):
            raise RuntimeError("ChunkQualityAnalyzer requires ChromaDB vectorstore")

        # 모든 청크 가져오기
        results = await asyncio.to_thread(
            self.vectorstore._collection.get,
            include=["documents", "metadatas"],
            limit=limit,
        )

        # 문서별로 그룹화
        doc_chunks: dict[str, list[dict]] = {}
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""
                doc_id = metadata.get("document_id", "unknown")

                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []

                doc_chunks[doc_id].append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                })

        # 각 문서별 통계 계산
        stats = {}
        for doc_id, chunks in doc_chunks.items():
            lengths = [len(chunk["content"]) for chunk in chunks]
            filename = chunks[0]["metadata"].get("filename", "unknown") if chunks else "unknown"

            stats[doc_id] = DocumentChunkStats(
                document_id=doc_id,
                filename=filename,
                chunk_count=len(chunks),
                avg_length=statistics.mean(lengths) if lengths else 0,
                min_length=min(lengths) if lengths else 0,
                max_length=max(lengths) if lengths else 0,
            )

        return stats

    def _build_histogram(self, lengths: list[int]) -> dict[str, int]:
        """청크 길이 히스토그램 생성."""
        histogram = {
            "0-200": 0,
            "200-400": 0,
            "400-600": 0,
            "600-800": 0,
            "800+": 0,
        }

        for length in lengths:
            if length < 200:
                histogram["0-200"] += 1
            elif length < 400:
                histogram["200-400"] += 1
            elif length < 600:
                histogram["400-600"] += 1
            elif length < 800:
                histogram["600-800"] += 1
            else:
                histogram["800+"] += 1

        return histogram

    def _calculate_special_char_ratio(self, chunks: list[dict]) -> float:
        """특수 문자 비율 계산 (비알파벳, 비한글, 비숫자)."""
        total_chars = 0
        special_chars = 0

        for chunk in chunks:
            content = chunk["content"]
            total_chars += len(content)

            for char in content:
                # 한글, 영문, 숫자가 아닌 문자 (공백 제외)
                if not (char.isalnum() or '\uac00' <= char <= '\ud7a3' or char.isspace()):
                    special_chars += 1

        return special_chars / total_chars if total_chars > 0 else 0.0

    def _count_duplicates(self, chunks: list[dict]) -> int:
        """완전 중복 청크 카운트 (해시 기반)."""
        seen_hashes = set()
        duplicates = 0

        for chunk in chunks:
            content_hash = hashlib.md5(chunk["content"].encode()).hexdigest()
            if content_hash in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(content_hash)

        return duplicates

    def _count_near_duplicates(self, chunks: list[dict]) -> int:
        """유사 중복 청크 카운트 (trigram Jaccard > 0.9).

        Note: For large datasets (>1000 chunks), uses sampling for approximate count.
        """
        import random

        near_duplicates = 0
        sample_chunks = chunks

        # Sample if dataset is large to avoid O(n^2) explosion
        if len(chunks) > 1000:
            sample_chunks = random.sample(chunks, 1000)

        # 모든 청크 쌍 비교 (O(n^2) - 샘플링으로 제한)
        for i in range(len(sample_chunks)):
            for j in range(i + 1, len(sample_chunks)):
                similarity = self._trigram_similarity(
                    sample_chunks[i]["content"],
                    sample_chunks[j]["content"]
                )
                if similarity > 0.9:
                    near_duplicates += 1

        return near_duplicates

    def _trigram_similarity(self, text1: str, text2: str) -> float:
        """Trigram Jaccard 유사도 계산."""
        def trigrams(text: str) -> set:
            text = text.lower()
            return {text[i:i+3] for i in range(len(text) - 2)}

        t1 = trigrams(text1)
        t2 = trigrams(text2)

        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0

        intersection = len(t1 & t2)
        union = len(t1 | t2)

        return intersection / union if union > 0 else 0.0

    def _calculate_semantic_completeness(self, chunks: list[dict]) -> float:
        """의미 완결성 계산 (문장 종결 부호로 끝나는 청크 비율)."""
        # 한국어 종결 어미 및 문장 종결 부호
        ending_patterns = [
            r'[.。!?]$',           # 마침표, 느낌표, 물음표
            r'[다요음임]$',         # 한국어 종결 어미
            r'(?:습니다|입니다|였습니다|됩니다|겠습니다|합니다)$',  # 정중한 종결 어미
        ]

        complete_chunks = 0
        for chunk in chunks:
            content = chunk["content"].strip()
            if any(re.search(pattern, content) for pattern in ending_patterns):
                complete_chunks += 1

        return complete_chunks / len(chunks) if chunks else 0.0
