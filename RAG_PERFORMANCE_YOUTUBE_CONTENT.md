# RAG 성능을 극적으로 올리는 10가지 실전 전략

> 유튜브 콘텐츠 스크립트 | 엔터프라이즈 RAG 시스템 실전 적용기
> 벤치마크 95.8% 달성까지의 여정 (120문항, 5개 데이터셋)

---

## 영상 정보

- **제목 후보**:
  - "RAG 성능 50%→95% 올린 10가지 실전 전략 (코드 공개)"
  - "기업용 RAG 시스템, 벤치마크 95.8% 달성한 비결"
  - "Naive RAG는 이제 그만! 프로덕션 RAG 10단계 진화"
- **예상 길이**: 25~35분
- **대상**: RAG 시스템을 구축하거나 개선하려는 개발자, AI 엔지니어
- **핵심 메시지**: "단순한 Vector Search → LLM으로는 실무에서 50%도 못 넘긴다. 10가지 전략을 하나씩 적용할 때마다 성능이 올라가는 걸 벤치마크로 증명한다."

---

## 인트로 (2분)

### 오프닝 훅

```
"RAG를 처음 만들면 누구나 이런 경험을 합니다.
'와, 문서 검색해서 답변 생성하니까 잘 되네!'
...라고 생각하다가 실제 벤치마크를 돌리면 50~60%대.

오늘은 제가 엔터프라이즈 RAG 시스템을 만들면서
95.8%까지 올린 10가지 실전 전략을 공유합니다.

이론이 아닙니다. 전부 코드와 벤치마크 결과가 있습니다."
```

### 벤치마크 환경 소개

```
- 120문항, 5개 데이터셋 (법률, 내부규정, 홍보자료, 경영공시, 업무보고서)
- 약 40,000개 청크 (ChromaDB)
- LLM: Qwen2.5 14B (Ollama 로컬)
- 임베딩: BAAI/bge-m3 (1024차원)
- 질문 유형: factual, inference, negative, multi-hop
```

### 최종 결과 미리보기 (시청 유지용)

| 데이터셋 | 문항 | 결과 |
|---------|------|------|
| 관련 법률 | 50 | **100%** |
| 내부규정 | 60 | **98.3%** |
| 홍보자료 | 20 | **100%** |
| 경영공시 | 20 | **100%** |
| 출장보고서 | 20 | **80%** |

---

## 전략 1: Hybrid Search — Vector + BM25 결합 (3분)

### 왜 Vector Search만으로는 부족한가

```
"Vector Search는 의미적 유사성에 강하지만,
정확한 키워드 매칭에는 약합니다.

예: '제37조'를 검색할 때
- Vector Search: '관련 조항'을 잘 찾지만 정확히 37조는 놓칠 수 있음
- BM25: 정확한 조문 번호를 키워드로 매칭

둘을 합치면? 의미도 잡고 키워드도 잡습니다."
```

### 구현 핵심

```python
# retriever.py — 하이브리드 검색 아키텍처

class HybridRetriever:
    """Vector + BM25 하이브리드 검색"""

    def __init__(self, vector_weight=0.7, bm25_weight=0.3):
        self.vector_weight = vector_weight   # 의미 검색 가중치
        self.bm25_weight = bm25_weight       # 키워드 검색 가중치

    async def retrieve(self, query, top_k=10):
        # 1단계: 벡터 검색 (3배 확장 풀)
        expanded_k = top_k * 3
        vector_results = await self._vector_search(query, expanded_k)

        # 2단계: BM25로 후보군 재채점 (전체 코퍼스 불필요!)
        bm25_results = self._bm25_score_candidates(query, vector_results)

        # 3단계: 가중 합산
        combined = self._merge_results(vector_results[:top_k], bm25_results)
        return combined
```

### 포인트 (화면에 강조)

```
핵심 트릭: BM25 전체 코퍼스 인덱스를 안 씁니다!

기존 방식: 전체 문서를 BM25 인덱스로 만들어서 따로 검색
→ 메모리 많이 먹고, 문서 추가할 때마다 재인덱싱 필요

우리 방식: Vector Search로 먼저 후보군(3x)을 뽑고,
그 후보군만 BM25로 재채점
→ 매 쿼리마다 작은 BM25 인덱스를 즉석 생성
→ 메모리 효율적, 문서 추가해도 즉시 반영
```

### 한국어 특화: 형태소 분석

```python
# Kiwi 형태소 분석기로 BM25 토크나이징
def _tokenize(self, text):
    """의미 있는 형태소만 추출"""
    tokens = []
    for token in self._kiwi.tokenize(text):
        # 명사(NN), 동사(VV), 형용사(VA), 어근(XR), 숫자(SN), 외국어(SL)
        if token.tag.startswith(('NN', 'VV', 'VA', 'XR', 'SN', 'SL')):
            tokens.append(token.form.lower())
    return tokens

# "산업안전관리법 제37조의 내용은?"
# → ['산업', '안전', '관리', '법', '제', '37', '조', '내용']
```

```
왜 형태소 분석이 중요한가?
영어: 공백으로 분리하면 됨
한국어: "품질관리기준에서는" → 단순 분리 불가
Kiwi가 "품질/관리/기준/에서/는"으로 정확히 분리
```

---

## 전략 2: Cross-Encoder Reranking (2분)

### 왜 필요한가

```
"Hybrid Search가 후보 10개를 뽑았다고 끝이 아닙니다.
Bi-encoder (bge-m3)는 빠르지만 정밀도가 떨어져요.

Cross-encoder는 질문과 문서를 함께 보고 관련도를 판단합니다.
느리지만 정확합니다.

전략: Bi-encoder로 넓게 뽑고 → Cross-encoder로 정밀 정렬"
```

### 2단계 검색 파이프라인

```
[사용자 질문]
    ↓
[1단계: Bi-Encoder (bge-m3)] — 빠르게 30개 후보 추출
    ↓
[2단계: Cross-Encoder (bge-reranker-v2-m3)] — 정밀 리랭킹
    ↓
[상위 5~8개만 LLM에 전달]
```

```python
# retriever.py — 리랭킹
async def _rerank(self, query, results):
    """Cross-Encoder로 정밀 리랭킹"""
    pairs = [(query, r.content) for r in results]
    scores = await asyncio.to_thread(self._reranker.predict, pairs)

    for i, result in enumerate(results):
        result.rerank_score = float(scores[i])
        result.score = float(scores[i])  # 리랭크 점수로 완전 교체

    return sorted(results, key=lambda r: r.score, reverse=True)
```

```
성능 팁: asyncio.to_thread()로 이벤트 루프 블로킹 방지!
Cross-Encoder predict()는 CPU 작업이라 비동기 쓰레드로 오프로딩
→ 다른 요청을 블로킹하지 않음
```

---

## 전략 3: HyDE — 가설 문서 임베딩 (3분)

### 핵심 아이디어

```
"사용자의 '질문'과 '답변이 있을 문서'는 의미 공간에서 멀리 있습니다.

예: 질문 '제37조의 내용은?'
   → 임베딩: 질문 형태의 벡터

하지만 찾고 싶은 건 '제37조 (벌칙) 다음 각 호의 어느 하나에...'
   → 임베딩: 법률 조문 형태의 벡터

질문 벡터 ≠ 답변 벡터 → 의미 갭 발생!"
```

### HyDE 해결책

```
HyDE (Hypothetical Document Embeddings)
1. LLM에게 "이 질문의 답변이 될 법한 가상 문서"를 생성시킴
2. 가상 문서를 임베딩
3. 가상 문서 임베딩으로 벡터 검색 실행

→ '질문 벡터'가 아닌 '답변 벡터'로 검색하는 효과!
```

### 구현

```python
# query_expander.py
class QueryExpander:
    async def expand_hyde(self, query: str) -> list[float]:
        """HyDE: 가설 문서 → 임베딩"""

        # Step 1: LLM이 가상 답변 생성
        prompt = f"""질문: {query}
        위 질문에 대한 답변이 포함된 문서의 내용을 작성하세요.
        실제 내용처럼 자연스럽게 작성하세요."""

        response = await self.llm.generate(prompt=prompt, temperature=0.7)
        hypothetical_doc = response.content

        # Step 2: 가상 답변을 임베딩
        embedding = await self.embedder.embed_query(hypothetical_doc)
        return embedding
```

### 고급: 원본 + HyDE 평균 벡터

```python
async def expand_hyde_with_original(self, query: str) -> list[float]:
    """HyDE + 원본 쿼리 임베딩 평균 (의도 보존)"""

    # 두 임베딩을 병렬로 생성
    hyde_emb, orig_emb = await asyncio.gather(
        self.expand_hyde(query),
        self.embedder.embed_query(query)
    )

    # 평균 + 정규화
    combined = (np.array(hyde_emb) + np.array(orig_emb)) / 2.0
    combined = combined / np.linalg.norm(combined)
    return combined.tolist()
```

```
실전 팁: 평균 벡터를 쓰는 이유
HyDE만 쓰면 LLM이 잘못된 가설을 생성했을 때 원래 의도에서 벗어남
원본 쿼리 벡터를 50% 섞어서 의도 보존 + 문서 유사성 둘 다 확보
```

---

## 전략 4: Multi-Query Retrieval (3분)

### 문제 상황

```
"하나의 질문으로 검색하면,
그 질문의 표현 방식에 맞는 문서만 나옵니다.

예: '감사규정과 상벌규정의 관계는?'
→ '감사규정'이 포함된 문서는 찾지만
  '상벌규정' 관련 문서는 놓칠 수 있음

특히 multi-hop 질문(여러 문서를 연결해야 하는 질문)에서 치명적!"
```

### 해결: 질문을 여러 관점으로 분해

```python
# multi_query.py
class MultiQueryRetriever:
    async def generate_queries(self, query: str) -> list[str]:
        """LLM으로 대안 질문 3개 생성"""
        prompt = f"""원본 질문: {query}
        이 질문을 다른 관점에서 3가지로 재작성하세요.
        JSON 배열로 반환: ["질문1", "질문2", "질문3"]"""

        response = await self.llm.generate(prompt=prompt)
        return json.loads(response.content)  # ["...", "...", "..."]

    async def retrieve(self, query: str, top_k=None):
        alt_queries = await self.generate_queries(query)
        all_queries = [query] + alt_queries  # 원본 포함 총 4개

        # 4개 쿼리를 병렬 실행!
        tasks = [self.retriever.retrieve(q, top_k) for q in all_queries]
        all_results = await asyncio.gather(*tasks)

        # 결과 병합 + 빈도 보너스
        return self._merge_multi_results(all_results)
```

### 결과 병합 전략: 빈도 보너스

```python
@staticmethod
def _merge_multi_results(result_sets):
    """여러 쿼리 결과 병합 — max score + 빈도 보너스"""
    chunk_map = {}      # chunk_id → best result
    chunk_frequency = {} # chunk_id → 몇 개 쿼리에서 발견됐나

    for result_set in result_sets:
        for r in result_set:
            chunk_frequency[r.id] = chunk_frequency.get(r.id, 0) + 1
            # 가장 높은 점수를 유지
            if r.id not in chunk_map or r.score > chunk_map[r.id].score:
                chunk_map[r.id] = r

    # 빈도 보너스: 여러 쿼리에서 발견된 청크 = 더 관련성 높음
    for chunk_id, result in chunk_map.items():
        freq = chunk_frequency[chunk_id]
        bonus = (freq - 1) * 0.05  # 추가 발견마다 +5%
        result.score = min(1.0, result.score + bonus)

    return sorted(chunk_map.values(), key=lambda x: x.score, reverse=True)
```

```
핵심 인사이트: 3개 쿼리에서 모두 등장한 청크 = 핵심 문서
빈도 보너스가 "우연히 높은 점수" vs "진짜 관련 문서"를 구분해줌
```

---

## 전략 5: Agentic RAG — 동적 전략 라우팅 (3분)

### 왜 모든 질문에 같은 전략을 쓰면 안 되는가

```
"안녕하세요" → RAG 검색이 필요할까? NO. 직접 답변하면 됨
"제37조 내용은?" → 기본 RAG로 충분
"인사규정과 상벌규정의 관계는?" → Multi-Query가 필요
"점검 절차를 상세히 설명해줘" → Deep Retrieval (더 많은 청크)

질문마다 최적 전략이 다릅니다!
```

### 4가지 전략 라우터

```python
# agentic.py
class AgenticRAGRouter:
    STRATEGY_DEFAULTS = {
        "standard_rag":    {"top_k": None, "temperature": None},
        "multi_query_rag": {"top_k": None, "temperature": 0.3},
        "direct_llm":      {"top_k": 0,    "temperature": 0.7},
        "deep_retrieval":  {"top_k": 30,   "temperature": 0.3},
    }

    async def route(self, query: str) -> RoutingDecision:
        """LLM이 질문을 분석하고 최적 전략 선택"""
        try:
            return await self._llm_route(query)      # 1차: LLM 판단
        except:
            return self._rule_based_route(query)      # 2차: 규칙 기반 폴백
```

### 규칙 기반 폴백 (LLM 없이도 동작)

```python
@staticmethod
def _rule_based_route(query: str) -> RoutingDecision:
    """LLM 라우팅 실패 시 키워드 기반 폴백"""

    # 인사/잡담 → 직접 응답
    if re.search(r'^(안녕|하이|감사|반갑)', query) and len(query) < 20:
        return RoutingDecision(strategy="direct_llm", ...)

    # 비교/분석 질문 → 멀티쿼리
    if re.search(r'(비교|차이|vs|종합|분석|요약)', query):
        return RoutingDecision(strategy="multi_query_rag", ...)

    # 상세/절차 질문 → 딥 리트리벌
    if re.search(r'(상세|구체적|절차|단계|방법)', query) and len(query) > 20:
        return RoutingDecision(strategy="deep_retrieval", ...)

    # 도메인 키워드 없음 → LLM 직접 응답
    if not re.search(r'(법|규정|조|설비|점검|안전)', query):
        return RoutingDecision(strategy="direct_llm", ...)

    # 기본값
    return RoutingDecision(strategy="standard_rag", ...)
```

```
설계 포인트: LLM 라우팅 + 규칙 폴백의 이중 구조

LLM 라우팅: 정확하지만 추가 LLM 호출 비용 발생 (100~200ms)
규칙 폴백: LLM 호출 없이 즉시 판단 (1ms 미만)

LLM이 실패하거나 느릴 때 자동으로 규칙 기반으로 전환
→ 어떤 상황에서도 시스템이 멈추지 않음
```

---

## 전략 6: Self-RAG — 자기반성 환각 탐지 (3분)

### RAG의 근본 문제: 환각

```
"RAG를 써도 환각(Hallucination)이 발생합니다.

왜? LLM이 컨텍스트에 없는 내용을 '아는 척' 생성하기 때문.
특히 법률/규정 도메인에서 이건 치명적입니다.
'제15조에 의하면...' 이라고 답변했는데 15조에 그런 내용이 없다면?"
```

### Self-RAG 프로세스

```
[1] 질문 + 컨텍스트 → LLM → 답변 생성
[2] 답변 + 컨텍스트 → LLM(평가자) → 근거성 채점
    - grounded? (컨텍스트에 근거하는가)
    - hallucination? (지어낸 내용이 있는가)
    - relevance? (질문에 맞는 답변인가)
[3-a] 통과 → 답변 반환
[3-b] 실패 → 더 엄격한 프롬프트로 재생성 → 재채점
```

```python
# self_rag.py
class SelfRAGEvaluator:
    STRICT_SUFFIX = """
    [중요 제약]
    - 위 컨텍스트에 명시적으로 있는 내용만 답변하세요.
    - 추론이나 추정을 하지 마세요.
    - 컨텍스트에서 직접 인용하여 답변하세요.
    - 확실하지 않은 내용은 '컨텍스트에서 확인할 수 없습니다'라고 답변하세요.
    """

    async def evaluate_and_retry(self, query, answer, context, ...):
        # 1차 채점
        grade = await self.grade_answer(query, answer, context)

        if grade.passed:
            return answer, metadata  # 통과!

        # 실패: 더 엄격한 프롬프트로 재시도
        for retry in range(self.max_retries):
            strict_system = system_prompt + self.STRICT_SUFFIX
            response = await llm.generate(
                prompt=user_prompt,
                system=strict_system,
                temperature=max(0.1, temperature * 0.5),  # 온도도 절반으로!
            )

            retry_grade = await self.grade_answer(query, response, context)
            if retry_grade.passed:
                return response, metadata  # 재시도 성공!

        return last_response, metadata  # 모든 재시도 실패 (메타데이터에 기록)
```

```
핵심: 재시도할 때 두 가지를 동시에 조정
1. 프롬프트: "명시적으로 있는 내용만 답변하세요" 추가
2. Temperature: 0.7 → 0.35 (절반으로 감소)
   → LLM이 더 보수적으로 답변하게 유도

메타데이터에 채점 결과가 남으므로
프론트엔드에서 "⚠️ AI가 자체 검증을 통과하지 못한 답변입니다" 표시 가능
```

---

## 전략 7: 쿼리 전처리 파이프라인 (3분)

### 사용자 입력은 완벽하지 않다

```
"실제 사용자 입력:
- '산안법 제37조' → '산업안전보건법 제37조' (약어)
- '전안법' → '전기안전관리법' (정식명칭)
- '품질관리기준' → '품질관리 기준' (띄어쓰기)

이런 입력을 그대로 검색하면 관련 문서를 못 찾습니다."
```

### 3단계 쿼리 전처리

```
[사용자 입력]
    ↓
[1단계: Query Correction — 오타/약어 교정]
    ↓
[2단계: Terminology Expansion — 전문용어 동의어 확장]
    ↓
[3단계: Source Type Auto-Detection — 검색 범위 자동 축소]
    ↓
[최적화된 쿼리 → Retriever]
```

#### 1단계: 도메인 유의어 교정

```python
# query_corrector.py — 도메인 전문 유의어 사전
DOMAIN_SYNONYMS = {
    "산안법":      "산업안전보건법",        # 법률 약어
    "전안법":      "전기안전관리법",        # 업계 약어
    "화관법":      "화학물질관리법",
    "고압법":      "고압가스 안전관리법",
    "LNG":        "액화천연가스(LNG)",     # 영어 약어 → 한국어 풀네임 + 약어 병기
    "CNG":        "압축천연가스(CNG)",
    "KS":         "한국산업표준(KS)",
    "ISO":        "국제표준화기구(ISO)",
    "품관법":      "품질관리법",
}
```

```
구현 트릭: Longest-First 매칭 + Placeholder 방식

"산업안전보건법" 안에 "안전보건"이 포함되어 있음
→ "안전보건" 패턴을 먼저 교정하면 "산업산업안전보건법"이 됨!

해결: 긴 패턴부터 교정 + 교정된 부분을 플레이스홀더로 보호
→ "산업안전보건법" 전체 매칭이 먼저 적용
→ 이미 교정된 부분은 다시 교정되지 않음
```

#### 2단계: 전문용어 동의어 확장

```python
# terminology.py — 도메인 전문용어 사전
# 검색 쿼리에 동의어를 추가하여 재현율(recall) 향상

# 예: "변압기 점검 주기" 입력 시
# → "변압기 전력변압기 점검 주기" 로 확장하여 검색
# → 원본 문서에 "전력변압기"로 표기되어 있어도 검색 가능
```

#### 3단계: Source Type 자동 탐지

```python
# chain.py — 질문 키워드로 검색 범위 자동 축소
def _auto_detect_filters(question, filters):
    """질문에서 데이터셋을 자동 판별 → ChromaDB 메타데이터 필터"""

    law_keywords = ["특별법", "시행령", "법 제", "부칙"]
    rule_keywords = ["내부규정", "인사규정", "감사규정", "여비규정"]
    brochure_keywords = ["홍보자료", "안내책자", "회사소개"]

    # "인사규정에서 퇴직금 지급 기준은?"
    # → source_type = "내부규정" 필터 자동 적용
    # → 39,000개 청크 중 19,000개만 검색 (약 50% 범위 축소)

    for kw in rule_keywords:
        if kw in question.lower():
            return {"source_type": "내부규정"}
```

```
성능 임팩트:
- 필터 없이 검색: 약 40,000개 청크에서 검색 → 노이즈 많음
- 필터 적용 후: 해당 카테고리만 검색 → 정밀도 대폭 향상

특히 '내부규정'과 '경영공시'가 각 19,000개로 비슷한 규모일 때
필터 없으면 공시 데이터가 규정 검색을 방해하는 문제 해결!
```

---

## 전략 8: 도메인 적응형 프롬프트 엔지니어링 (3분)

### 하나의 프롬프트로는 안 된다

```
"법률 질문과 기술 매뉴얼 질문은 답변 스타일이 완전히 다릅니다.

법률: '제37조에 의거, 위반 시 5년 이하의 징역...'
기술: '점검 절차는 다음과 같습니다. 1단계...'
일반: 'A사는 1986년에 설립된 공공기관으로...'

하나의 시스템 프롬프트로 세 가지를 다 커버하기 어렵습니다."
```

### 3가지 자동 프롬프트 선택

```python
# prompt.py — 검색된 문서에서 자동으로 도메인 판별
@staticmethod
def detect_document_type(context_chunks):
    """검색된 청크의 메타데이터에서 문서 유형 자동 판별"""
    legal_score = 0
    technical_score = 0
    general_score = 0

    for chunk in context_chunks:
        source_type = chunk["metadata"].get("source_type", "")
        content = chunk["content"]

        if source_type in ("법률", "내부규정", "정관"):
            legal_score += 3
        if re.search(r'제\d+조', content):  # '제37조' 패턴
            legal_score += 1
        if source_type in ("홍보자료", "업무보고서", "경영공시"):
            general_score += 3

    if legal_score > technical_score and legal_score > general_score:
        return "legal"       # → rag_legal_system 프롬프트
    elif technical_score > legal_score:
        return "technical"   # → rag_technical_system 프롬프트
    return "general"         # → rag_system 프롬프트
```

### 모델 크기 인식 프롬프팅

```python
# 7B 모델: 장황한 답변 경향 → 극단적 간결성 지시
# 14B 모델: 중간 수준 → 유형별 길이 가이드
# 70B+ 모델: 추가 지시 불필요

@staticmethod
def _get_conciseness_suffix(model_hint):
    if "7b" in model_hint.lower():
        return """
        [추가 지시 - 반드시 따르세요]
        - 반드시 1~3문장으로만 답변하세요.
        - 법 조문의 원문을 그대로 인용하세요.
        - 추가 해석이나 부연 설명을 절대 덧붙이지 마세요.
        """
    elif "14b" in model_hint.lower():
        return """
        [추가 지시 - 반드시 따르세요]
        - 사실(factual) 질문: 50~150자로 간결하게 답변하세요.
        - 부정(negative) 질문: 100~250자로, 이유를 설명하세요.
        - 핵심 사실만 답변하고, 원문 표현을 그대로 사용하세요.
        """
    return ""  # 대형 모델은 추가 지시 불필요
```

### 카테고리별 Few-Shot 예시

```
YAML로 관리하는 few-shot 예시 (총 8개):
- factual 2개: "제37조의 벌칙은?" → "5년 이하의 징역..."
- negative 2개: "~에 대한 규정이 있나?" → "확인되지 않습니다..."
- inference 2개: "이 규정의 의미는?" → "~를 의미합니다..."
- multi_hop 2개: "A규정과 B규정의 관계?" → "~에 따르면..."

법률 도메인 → legal_examples (조문 인용 스타일)
기술 도메인 → technical_examples (절차 설명 스타일)
일반 도메인 → general_examples (정보 전달 스타일)
```

---

## 전략 9: 응답 품질 검증 + 자동 재시도 (2분)

### LLM 출력은 항상 깨끗하지 않다

```
"7B~14B 모델을 쓰면 이런 문제가 자주 발생합니다:
1. 깨진 출력: 중간에 잘린 답변
2. 중국어 누출: 중국어 기반 모델(Qwen)이 가끔 중국어로 답변
3. Q&A 형식 반복: 'Q: 질문 A: 답변' 패턴을 그대로 출력
4. 출처 태그 노출: '[출처: 문서명]' 같은 내부 태그가 그대로 노출"
```

### 검증 + 자동 재시도 + 후처리

```python
# chain.py — 검증 → 재시도 → 후처리 3단계

# 1. 생성 + 검증 (최대 3회)
for attempt in range(max_retries + 1):
    response = await llm.generate(prompt=user_prompt, system=system_prompt)
    if self._is_valid_response(response.content):
        break  # 유효한 응답이면 즉시 사용
    logger.warning("Invalid response (attempt %d), retrying", attempt + 1)

# 2. 유효성 검증 함수
@staticmethod
def _is_valid_response(text):
    stripped = text.strip()
    if len(stripped) < 15:           return False  # 너무 짧음
    content_chars = re.sub(r'[\s\-\n•·]', '', stripped)
    if len(content_chars) < 10:      return False  # 실질 내용 없음
    chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', stripped)) / max(len(stripped), 1)
    if chinese_ratio > 0.3:          return False  # 중국어 30% 초과
    return True

# 3. 후처리 (아티팩트 제거)
_RE_QA_PATTERN = re.compile(r'^Q[:：].*?(?:A[:：])\s*', re.DOTALL)
_RE_A_PREFIX = re.compile(r'^(A[:：]|답변[:：])\s*')
_RE_SOURCE_TAG = re.compile(r'\[출[처처][^\]]*\]')
_RE_CJK_LEAKAGE = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+[^\n]*$', re.MULTILINE)
```

```
성능 팁: 정규식을 모듈 레벨에서 프리컴파일!

매 쿼리마다 re.compile()하면 의미 없는 CPU 낭비
6개 패턴을 한 번만 컴파일하고 재사용 → 응답 시간 수ms 절약
```

---

## 전략 10: Multi-Hop 자동 탐지 + 동적 파라미터 (2분)

### Multi-Hop 질문이란?

```
"여러 문서를 연결해야 답변할 수 있는 질문:

'감사규정과 상벌규정의 연계 사항은?'
→ 감사규정 문서 + 상벌규정 문서 둘 다 필요

'인사규정에서 채용부터 퇴직까지의 전체 프로세스는?'
→ 채용 관련 조항 + 복무 관련 조항 + 퇴직 관련 조항

일반 RAG: top_k=10 → 한 규정의 청크만 가져옴
Multi-hop: top_k=30 + Multi-Query → 여러 규정의 청크를 골고루 가져옴"
```

### 자동 탐지 + 파라미터 동적 조정

```python
# chain.py — Multi-hop 자동 탐지
@staticmethod
def _detect_multi_hop(question):
    """Multi-hop 질문 키워드 자동 탐지"""
    multi_hop_keywords = [
        # 교차 참조 마커
        "연계", "연결되", "관계가 있", "연관",
        # 프로세스/절차 관련
        "프로세스를", "절차를", "과정을", "단계를",
        # 복수 규정 참조
        "규정과", "규정 간", "규정들", "각각",
        # 비교/대조
        "차이점", "비교", "어떻게 다른",
        # 종합 추론
        "두 자료", "두 문서", "종합하면",
    ]
    return any(kw in question.lower() for kw in multi_hop_keywords)

# 탐지 시 자동 파라미터 조정
if self._detect_multi_hop(question):
    agentic_use_multi_query = True  # Multi-Query 강제 활성화
    agentic_top_k = 30              # 검색 범위 3배 확대
    effective_rerank_top_n = 8      # 리랭크 후 상위 8개 사용
```

```
결과: Multi-hop 질문의 정답률이 크게 향상
- Before: 단일 규정 청크만 검색 → 부분적 답변
- After: 관련 규정 3~4개의 청크가 골고루 검색 → 종합 답변
```

---

## 보너스: 시스템 레벨 성능 최적화 (3분)

### 이건 RAG 전략은 아니지만, 없으면 서비스가 안 됩니다

```
"아무리 좋은 RAG 전략도
응답 시간이 10초면 아무도 안 씁니다.

시스템 레벨 최적화가 없으면
RAG 전략의 성능 향상이 사용자 경험으로 이어지지 않습니다."
```

#### 캐싱 전략 (2-Level)

```python
# 1. 쿼리 캐시: 동일 질문 → 즉시 응답 (TTL 60초)
cache_key = "rag:query:" + sha256(query + mode + filters)
cached = await cache.get(cache_key)
if cached:
    return RAGResponse(**cached)  # LLM 호출 없이 즉시 반환

# 2. 검색 캐시: 동일 검색 조건 → 벡터 검색 스킵 (TTL 120초)
cache_key = "retriever:search:" + sha256(query + top_k + filters)
cached = await cache.get(cache_key)
if cached:
    return [RetrievalResult(**r) for r in cached]  # 임베딩/검색 스킵
```

```
2레벨 캐시의 효과:
- 동일 질문: LLM 호출도 스킵 → 응답 시간 <10ms
- 유사 질문 (다른 모드): 검색만 스킵, LLM만 실행 → 50% 절약
```

#### 비동기 아키텍처

```python
# BM25: CPU 바운드 → 별도 스레드
bm25_results = await asyncio.to_thread(self._bm25_score_candidates, query, candidates)

# Cross-Encoder: CPU 바운드 → 별도 스레드
scores = await asyncio.to_thread(reranker.predict, pairs)

# Multi-Query: I/O 바운드 → asyncio.gather로 병렬 실행
all_results = await asyncio.gather(*[retriever.retrieve(q) for q in queries])

# 가드레일 로그: 크리티컬 패스에서 분리
def _log_fire_and_forget(self, direction, rule_id, ...):
    asyncio.ensure_future(self._log(...))  # await 없이 백그라운드 실행
```

#### DB 커넥션 최적화

```python
# AsyncSQLiteManager — WAL 모드 + 커넥션 풀
class AsyncSQLiteManager:
    async def _ensure_initialized(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")      # 동시 읽기 허용
            await db.execute("PRAGMA synchronous=NORMAL")     # 쓰기 성능 향상
            await db.execute("PRAGMA busy_timeout=5000")      # 락 대기 5초
```

---

## 전체 파이프라인 한눈에 보기 (2분)

```
[사용자 질문 입력]
    │
    ├─ 가드레일: 입력 검사 (프롬프트 인젝션 탐지)
    │
    ├─ Query Correction: 오타/약어 교정
    │    "산안법" → "산업안전보건법"
    │
    ├─ Terminology Expansion: 전문용어 동의어 확장
    │    "변압기" → "변압기 전력변압기"
    │
    ├─ Agentic Routing: 질문 분석 → 최적 전략 선택
    │    ├─ standard_rag / multi_query / direct_llm / deep_retrieval
    │    └─ Multi-hop 자동 탐지
    │
    ├─ Source Type Auto-Detection: 검색 범위 자동 축소
    │    "인사규정에서..." → source_type = "내부규정"
    │
    ├─ [HyDE] 가설 문서 임베딩 생성 (선택적)
    │
    ├─ [Multi-Query] 대안 질문 생성 + 병렬 검색 (선택적)
    │
    ├─ Hybrid Search: Vector (bge-m3) + BM25 (Kiwi 형태소)
    │
    ├─ Cross-Encoder Reranking (bge-reranker-v2-m3)
    │
    ├─ 도메인 프롬프트 선택: legal / technical / general
    │    + 모델 크기 인식 간결성 지시
    │    + Few-shot 예시 (도메인별 8개)
    │
    ├─ LLM 생성 + 유효성 검증 + 자동 재시도
    │
    ├─ Self-RAG: 근거성 채점 + 환각 탐지
    │    실패 시 → 엄격 프롬프트로 재생성
    │
    ├─ 후처리: Q&A 패턴/출처태그/CJK 누출 제거
    │
    ├─ 가드레일: 출력 검사 (PII 마스킹, 부적절 콘텐츠)
    │
    └─ [캐시 저장] → 응답 반환
```

---

## 벤치마크 검증 시스템 (2분)

### 골든 데이터셋 기반 자동 평가

```
"이 모든 전략이 실제로 효과가 있는지 어떻게 확인할까요?
120문항의 골든 데이터셋 + 자동 평가 시스템입니다."
```

```python
# evaluator.py — 복합 평가 메트릭
class AnswerEvaluator:
    """3가지 메트릭 + 카테고리별 가중치"""

    # 종합 점수 = 0.65 × 의미유사도 + 0.20 × ROUGE-L + 0.15 × ROUGE-1
    # → 한국어 법률 QA에서는 의미 유사도 비중을 높임
    #   (같은 법률 개념이 다양한 표현으로 기술되므로)

    CATEGORY_WEIGHTS = {
        "factual":   {"semantic": 0.55, "rouge_l": 0.30, "rouge_1": 0.15},
        "inference":  {"semantic": 0.70, "rouge_l": 0.15, "rouge_1": 0.15},
        "negative":   {"semantic": 0.80, "rouge_l": 0.10, "rouge_1": 0.10},
        "multi_hop": {"semantic": 0.65, "rouge_l": 0.20, "rouge_1": 0.15},
    }

    # 추가: 장황한 답변 패널티 (2.5배 초과 시 감점)
    if actual_len > expected_len * 2.5:
        length_penalty = max(0.7, 1.0 - (ratio - 2.5) * 0.04)

    # 추가: 부정 질문 오탐 패널티
    # 골든 답변이 "확인되지 않습니다"인데 실제 답변이 긍정적 → 0.3배 감점
```

### 한국어 ROUGE 최적화

```python
class _KoreanCharTokenizer:
    """문자 수준 한국어 토크나이저"""
    def tokenize(self, text):
        # 한국어는 교착어 → 단어 수준 ROUGE가 무의미
        # 문자 수준 매칭이 훨씬 정확
        text = re.sub(r"\[출처[^\]]*\]", "", text)   # 출처 태그 제거
        text = re.sub(r"^[AQ][:：]\s*", "", text)     # Q&A 접두어 제거
        return [c for c in text if re.match(r"[\uAC00-\uD7A3a-zA-Z0-9]", c)]
```

---

## 정리 + 실전 적용 가이드 (2분)

### 전략별 난이도와 효과

| # | 전략 | 구현 난이도 | 성능 효과 | 우선 적용 |
|---|------|-----------|----------|----------|
| 1 | Hybrid Search | ★★☆ | ★★★★ | 1순위 |
| 2 | Reranking | ★☆☆ | ★★★☆ | 1순위 |
| 3 | HyDE | ★★☆ | ★★★☆ | 3순위 |
| 4 | Multi-Query | ★★☆ | ★★★☆ | 2순위 |
| 5 | Agentic Routing | ★★★ | ★★☆☆ | 3순위 |
| 6 | Self-RAG | ★★★ | ★★★★ | 2순위 |
| 7 | 쿼리 전처리 | ★★☆ | ★★★★ | 1순위 |
| 8 | 도메인 프롬프트 | ★★☆ | ★★★★★ | 1순위 |
| 9 | 응답 검증+재시도 | ★☆☆ | ★★★☆ | 1순위 |
| 10 | Multi-Hop 탐지 | ★★☆ | ★★★☆ | 2순위 |

### 추천 적용 순서

```
[STEP 1 — 기본기] 1주일이면 적용 가능
  ✅ Hybrid Search (Vector + BM25)
  ✅ Reranking
  ✅ 도메인 프롬프트 + Few-Shot
  ✅ 응답 검증 + 후처리
  → 이것만으로 70~80%대 진입

[STEP 2 — 고급] 2주일 추가
  ✅ Multi-Query Retrieval
  ✅ Self-RAG
  ✅ 쿼리 전처리 (교정 + 용어 확장)
  ✅ Multi-Hop 자동 탐지
  → 85~95%대 진입

[STEP 3 — 엔터프라이즈] 필요 시
  ✅ HyDE
  ✅ Agentic Routing
  ✅ Source Type 자동 필터링
  ✅ 캐싱 + 비동기 최적화
  → 95%+ 안정화
```

### 클로징

```
"핵심은 단 하나입니다.
RAG는 하나의 기술이 아니라 파이프라인입니다.

검색, 프롬프트, 생성, 검증 — 각 단계마다
작은 개선을 쌓으면 전체 성능이 극적으로 올라갑니다.

오늘 소개한 10가지 전략의 코드는
영상 설명란의 GitHub에서 확인하실 수 있습니다.

여러분의 RAG 시스템은 몇 %인가요?
댓글로 알려주세요. 궁금한 전략이 있으면 딥다이브 영상도 만들겠습니다."
```

---

## 썸네일 / 제목 후보

### 썸네일 텍스트
- "RAG 50% → 95.8%"
- "10가지 전략"
- 그래프 이미지 (우상향)

### 제목 후보 (클릭률 최적화)
1. "RAG 성능 50%→95% 올린 10가지 실전 전략 (전체 코드 공개)"
2. "Naive RAG는 쓰레기다 | 프로덕션 RAG 10단계 완전정복"
3. "기업에서 RAG 만들면서 겪은 삽질 + 해결법 (벤치마크 95.8%)"
4. "ChatGPT로도 못 하는 것: 엔터프라이즈 RAG 성능 극한까지 올리기"

### 태그
RAG, LLM, 벡터검색, ChromaDB, BM25, HyDE, Self-RAG, Multi-Query, 한국어NLP, 기업AI, 프롬프트엔지니어링, 파이썬, FastAPI, Qwen

---

## 부록: 화면 구성 가이드

### 코드 보여줄 때
- VS Code 화면 캡처 (다크 테마)
- 핵심 코드만 하이라이트 (전체 파일 X)
- 주석으로 한국어 설명 추가

### 아키텍처 다이어그램
- Excalidraw 또는 draw.io로 그린 파이프라인 도식
- 각 전략이 파이프라인의 어디에 위치하는지 시각화

### 벤치마크 결과
- 전략 하나 추가할 때마다 점수 변화 그래프
- Before/After 비교 테이블

### 실제 질문-답변 데모
- 터미널에서 실시간 질의
- 검색된 소스, 신뢰도 점수, 라우팅 결과 등 메타데이터 함께 표시
