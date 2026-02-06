# flux-rag PRD

**Product Requirements Document**
**Version:** 1.0
**Last Updated:** 2026-02-05
**Status:** Draft

---

## 1. 프로젝트 개요

### 1.1 프로젝트 정보

- **프로젝트명**: flux-rag
- **목적**: 한국가스기술공사 생성형 AI 플랫폼 구축 사업 제안서 시연용 RAG 프레임워크
- **기간**: 1개월 (2026년 2월 ~ 2026년 3월)
- **범위**: Step1~Step5 전 범위 커버
  - Step 1: 환경설정 및 기반 구축
  - Step 2: 데이터 자산화
  - Step 3: RAG 파이프라인 최적화
  - Step 4: 포털 UI 구현
  - Step 5: 에이전트 모듈 구현
- **특이사항**: 에이전트 모듈 최대 구현 (MCP, Agent Builder, Tool 등록 포함)

### 1.2 운영 환경

#### AI LLM 서버
- **CPU**: Intel 6세대 64Core x2
- **GPU**: NVIDIA H200 NVL x4
- **RAM**: 1TB
- **Storage**: 1.92TB NVMe SSD x4
- **용도**: LLM 추론, 임베딩 생성, 모델 서빙

#### DB 서버 (x2)
- **CPU**: Intel 5세대 16Core x2
- **GPU**: NVIDIA L40S 48GB x4
- **RAM**: 256GB
- **Storage**: 960GB NVMe SSD x2
- **용도**: 벡터 DB, 메타데이터 DB, 로그 저장

#### 네트워크 환경
- **OS**: Ubuntu Linux
- **망분리**: 인터넷 차단 환경
- **보안**: 내부 네트워크만 접근 가능

### 1.3 프로젝트 제약사항

- 망분리 환경으로 인한 외부 API 제한 (OCR 등 사전 구성 필요)
- 1개월 내 전체 Step 완료 필요
- 시연용이지만 실제 운영 수준의 품질 요구
- 한국가스기술공사 특화 요구사항 반영 필수

---

## 2. 목표 및 비전

### 2.1 프로젝트 목표

#### 주요 목표
1. **RFP 요구사항 완전 커버**: 51개 요구사항 전체 구현
2. **고품질 RAG 시스템**: 정확도 90% 이상 달성
3. **확장 가능한 아키텍처**: 향후 기능 추가 용이
4. **에이전트 플랫폼**: MCP 기반 확장 가능 에이전트 시스템

#### 비즈니스 목표
- 한국가스기술공사 제안서 수주
- 생성형 AI 플랫폼 구축 역량 증명
- 향후 유사 프로젝트 레퍼런스 확보

### 2.2 RFP 요구사항 커버리지

총 51개 요구사항:
- **ECR (환경 요구사항)**: 4개
- **SFR (시스템 기능 요구사항)**: 18개
- **INR (인터페이스 요구사항)**: 7개
- **DAR (데이터 요구사항)**: 6개
- **TER (기술 요구사항)**: 1개
- **SER (시스템 환경 요구사항)**: 7개
- **QUR (품질 요구사항)**: 4개
- **COR (제약 요구사항)**: 1개
- **PMR (프로젝트 관리 요구사항)**: 1개
- **PSR (프로세스 요구사항)**: 2개

### 2.3 성능 목표

#### RAG 품질 지표 (QUR-004)
- **Recall**: ≥ 90%
- **Precision**: ≥ 90%
- **Semantic Similarity**: ≥ 90%
- **측정 방식**: 100-200쌍 Q&A 정답셋 기반 벤치마크

#### 시스템 성능 목표
- **응답 시간**: 평균 3초 이내 (스트리밍 시작까지 1초 이내)
- **동시 사용자**: 50명 이상
- **문서 처리**: 1,000페이지 PDF 10분 이내 처리
- **검색 성능**: 10,000개 청크 기준 100ms 이내

### 2.4 데이터 규모

#### 초기 데이터셋 (DAR-003)
- **문서 수**: 약 2,000개
- **총 용량**: 약 35GB
- **문서 유형**:
  - PDF (기술 문서, 보고서)
  - HWP (공문, 내부 문서)
  - XLSX (데이터 시트, 통계)
  - PPTX (발표 자료)
  - DOCX (일반 문서)
  - TXT (텍스트 파일)

#### 예상 청크 규모
- **청크 수**: 약 100,000개 (문서당 평균 50개)
- **청크 크기**: 500-1,000자 (10% 오버랩)
- **임베딩 벡터**: 약 100,000개 (차원: 1024)

---

## 3. 기술 스택

### 3.1 전체 기술 스택

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| **언어** | Python | 3.11+ | Backend 주 언어 |
| **프론트엔드** | React | 18+ | TypeScript, SPA |
| **UI 프레임워크** | Material-UI | 5.x | 반응형 디자인 |
| **백엔드 API** | FastAPI | 0.109+ | REST + SSE 스트리밍 |
| **LLM 서빙** | vLLM | 0.6+ | 고성능 LLM 추론 엔진 |
| **패키지 관리** | Poetry | 1.7+ | Python 의존성 관리 |
| **번들러** | Vite | 5.x | 프론트엔드 빌드 |

### 3.2 AI/ML 스택

#### LLM (Large Language Model)
- **시연 환경**: vLLM on RunPod (클라우드 GPU 인스턴스)
  - 장점: 고성능 추론, 배치 처리, 오픈소스 모델 자유도
  - 단점: 클라우드 비용
- **운영 환경**: vLLM on-premise (H200 NVL x4)
  - 장점: 망분리 환경 적합, 최고 성능, 비용 없음
  - 단점: 초기 구축 필요
- **개발/테스트**: Ollama (로컬 개발)
  - 장점: 빠른 셋업, 개발 편의성
  - 단점: 성능 제한 (프로덕션 부적합)
- **클라우드 대안**: OpenAI GPT-4 / Anthropic Claude
  - 장점: 고품질 응답, 관리 불필요
  - 단점: 비용, 망분리 환경 제약
- **LLM 추상화 계층**: vLLM Provider 포함하여 모델 교체 용이
  - vLLM은 OpenAI-compatible API 제공 (동일한 클라이언트 인터페이스)

#### 임베딩 모델
- **주 모델**: sentence-transformers (bge-m3)
  - 한국어 특화 성능
  - 다국어 지원
  - 차원: 1024
- **대안**: OpenAI text-embedding-3-large
  - 고품질 임베딩
  - API 기반

#### 벡터 데이터베이스
- **ChromaDB**
  - 로컬 persistent storage
  - Python 네이티브 통합
  - HNSW 인덱스 (고속 검색)
  - 메타데이터 필터링 지원

### 3.3 문서 처리 스택

#### OCR (광학 문자 인식)
- **Upstage Document Parse API**
  - 스캔 PDF 처리
  - 표 구조 인식
  - 신뢰도 점수 제공
  - 한국어 특화

#### 문서 파서
- **PDF**: PyMuPDF (fitz) - 고속, 텍스트/이미지 추출
- **HWP**: pyhwp → Markdown 변환 (기존 코드 참조)
- **Excel**: openpyxl - 표 구조 유지
- **PowerPoint**: python-pptx - 슬라이드별 추출
- **Word**: python-docx - 문단/표 추출
- **TXT**: 직접 읽기

### 3.4 검색 및 랭킹

#### 검색 전략
- **Hybrid Search**: BM25 (키워드) + 벡터 (시맨틱)
  - BM25: 정확한 용어 매칭
  - 벡터: 의미적 유사도
  - 가중치: 조정 가능 (기본 0.3:0.7)

#### Re-ranking
- **flashrank**: 경량 Re-ranking 모델
  - 검색 결과 재정렬
  - 관련성 점수 향상
  - 추론 속도 빠름

### 3.5 데이터 저장소

#### 메타데이터 및 이력
- **SQLite**: 개발/시연용
  - 대화 이력
  - 문서 메타데이터
  - 사용자 피드백
  - 로그
- **마이그레이션 경로**: PostgreSQL (운영 시)

#### 벡터 저장소
- **ChromaDB**: 임베딩 벡터
  - 영구 저장 (파일 기반)
  - 메타데이터 인덱싱
  - 컬렉션 분리 (knowledge_base, golden_data)

### 3.6 벡터DB 마이그레이션 경로

**단계별 전략**

#### 시연/개발 단계
- **ChromaDB** (로컬, 파일 기반)
  - 장점: 빠른 프로토타이핑, 설치 간편, Python 네이티브
  - 단점: 단일 노드, GPU 가속 미지원
  - 용도: 초기 개발, 알고리즘 검증, 소규모 데이터 테스트

#### 운영 단계
- **Milvus** (분산 벡터DB, L40S GPU 활용)
  - 장점: 대규모 확장성, GPU 가속 인덱싱, 엔터프라이즈급 안정성
  - 특징: DB 서버 L40S x4 활용하여 벡터 검색 가속화
  - 용도: 프로덕션 배포, 수백만 벡터 처리

#### 마이그레이션 전략
1. **동일 임베딩 모델 사용**: bge-m3 (1024차원) 고정 → 벡터 호환성 보장
2. **추상화 계층 유지**: `VectorStore` 인터페이스로 DB 교체 용이
3. **Export/Import 스크립트**: ChromaDB → Milvus 벡터 마이그레이션 도구
4. **점진적 전환**: 병렬 실행 기간 동안 결과 비교 검증

#### 구현 예시
```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors, metadata): pass

    @abstractmethod
    def search(self, query_vector, top_k): pass

class ChromaStore(VectorStore): ...
class MilvusStore(VectorStore): ...

# 설정 파일로 교체
vector_store = create_vector_store(config.vector_db_type)  # "chroma" or "milvus"
```

### 3.7 설정 관리

- **pydantic-settings**: 타입 안전 설정
- **.env 파일**: 환경별 설정
- **계층적 설정**: 기본값 → 환경변수 → .env 파일

### 3.7 개발 도구

#### 코드 품질
- **Linting**: Ruff (Python), ESLint (TypeScript)
- **Formatting**: Ruff (Python), Prettier (TypeScript)
- **Type Checking**: mypy (Python), tsc (TypeScript)

#### 테스트
- **Backend**: pytest, pytest-asyncio
- **Frontend**: Vitest, React Testing Library

#### 컨테이너화
- **Docker**: 개발 환경 일관성
- **docker-compose**: 멀티 컨테이너 오케스트레이션

---

## 4. 시스템 아키텍처

### 4.1 프로젝트 구조

```
flux-rag/
├── README.md
├── docker-compose.yml
├── .env.example
│
├── docs/
│   ├── PRD.md                    # 본 문서
│   ├── API.md                    # API 상세 문서
│   └── DEPLOYMENT.md             # 배포 가이드
│
├── backend/
│   ├── pyproject.toml            # Poetry 설정
│   ├── poetry.lock
│   ├── .env
│   │
│   ├── config/
│   │   └── settings.py           # 전역 설정
│   │
│   ├── core/                     # 핵심 추상화 계층
│   │   ├── llm/
│   │   │   ├── base.py           # LLM 추상 인터페이스
│   │   │   ├── ollama.py         # Ollama 구현
│   │   │   ├── openai.py         # OpenAI 구현
│   │   │   └── anthropic.py      # Anthropic 구현
│   │   ├── embeddings/
│   │   │   ├── base.py           # 임베딩 추상 인터페이스
│   │   │   ├── local.py          # 로컬 sentence-transformers
│   │   │   └── openai.py         # OpenAI 임베딩
│   │   └── vectorstore/
│   │       ├── base.py           # 벡터 DB 추상 인터페이스
│   │       └── chroma.py         # ChromaDB 구현
│   │
│   ├── pipeline/                 # 데이터 파이프라인
│   │   ├── loader.py             # 다중 포맷 문서 로더
│   │   ├── chunker.py            # 시맨틱 청킹
│   │   ├── metadata.py           # 메타데이터 추출/태깅
│   │   ├── ocr.py                # OCR 처리 (Upstage)
│   │   └── ingest.py             # 통합 수집 파이프라인
│   │
│   ├── rag/                      # RAG 엔진
│   │   ├── retriever.py          # 하이브리드 검색 + Re-ranking
│   │   ├── prompt.py             # 프롬프트 엔지니어링
│   │   ├── chain.py              # LLM 체인 (질의-응답)
│   │   └── quality.py            # 품질 평가 (Golden Data)
│   │
│   ├── agent/                    # 에이전트 플랫폼
│   │   ├── router.py             # 질문 유형 라우팅
│   │   ├── planner.py            # 복합 질문 분해
│   │   ├── memory.py             # 대화 메모리
│   │   ├── mcp/                  # Model Context Protocol
│   │   │   ├── server.py         # MCP 서버
│   │   │   └── tools/            # MCP 도구들
│   │   │       ├── search.py
│   │   │       └── calculator.py
│   │   ├── tool_registry.py      # 도구 등록 시스템
│   │   └── builder.py            # Agent Builder UI 백엔드
│   │
│   ├── api/
│   │   ├── main.py               # FastAPI 앱
│   │   ├── schemas.py            # Pydantic 스키마
│   │   └── routes/
│   │       ├── chat.py           # 채팅 엔드포인트
│   │       ├── documents.py      # 문서 관리
│   │       ├── admin.py          # 관리자 설정
│   │       ├── feedback.py       # 피드백/Golden Data
│   │       ├── agents.py         # 에이전트 관리
│   │       └── monitoring.py     # 모니터링/통계
│   │
│   ├── prompts/                  # 프롬프트 템플릿
│   │   ├── system.yaml           # 시스템 프롬프트
│   │   └── few_shot.yaml         # Few-shot 예시
│   │
│   └── data/                     # 로컬 데이터
│       ├── sample/               # 샘플 문서
│       ├── chroma_db/            # ChromaDB 영구 저장소
│       └── sqlite.db             # SQLite DB
│
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    │
    └── src/
        ├── main.tsx
        ├── App.tsx
        │
        ├── pages/
        │   ├── ChatPage.tsx          # 메인 채팅 인터페이스
        │   ├── DocumentsPage.tsx     # 문서 관리
        │   ├── AdminPage.tsx         # 관리자 포털
        │   ├── MonitorPage.tsx       # 모니터링 대시보드
        │   └── AgentBuilderPage.tsx  # 에이전트 빌더
        │
        ├── components/
        │   ├── chat/
        │   │   ├── ChatMessage.tsx
        │   │   ├── SourceViewer.tsx
        │   │   └── ModelSelector.tsx
        │   ├── documents/
        │   │   ├── FileUploader.tsx
        │   │   └── DocumentList.tsx
        │   └── admin/
        │       ├── ModelConfig.tsx
        │       └── PromptEditor.tsx
        │
        └── api/
            └── client.ts             # API 클라이언트
```

### 4.2 데이터 플로우

#### 4.2.1 문서 수집 파이프라인

```
┌─────────────┐
│ 파일 업로드  │ (PDF, HWP, XLSX, DOCX, PPTX, TXT)
└──────┬──────┘
       ↓
┌─────────────────────────────────────────┐
│ Loader (pipeline/loader.py)             │
│ - 포맷 감지                              │
│ - 적절한 파서 선택                       │
│ - 텍스트 추출                            │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ OCR (pipeline/ocr.py)                   │
│ - 스캔 문서 감지 (이미지 비율 > 50%)     │
│ - Upstage API 호출                       │
│ - 신뢰도 점수 확인                       │
│ - 문맥 보정 (LLM)                        │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Chunker (pipeline/chunker.py)           │
│ - 시맨틱 청킹 (500-1000자)               │
│ - 10% 오버랩                             │
│ - 문단/섹션 경계 인식                    │
│ - 표 구조 보존                           │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Metadata Extractor (pipeline/metadata.py)│
│ - 날짜 추출 (정규식)                     │
│ - 부서명 추출 (NER)                      │
│ - 문서 유형 분류                         │
│ - 태그 자동 생성                         │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Embedder (core/embeddings/)             │
│ - bge-m3 모델 (1024차원)                 │
│ - 배치 처리 (32개씩)                     │
│ - GPU 가속                               │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ ChromaDB 저장 (core/vectorstore/)       │
│ - 벡터 + 메타데이터 저장                 │
│ - HNSW 인덱스 생성                       │
│ - 컬렉션: knowledge_base                 │
└─────────────────────────────────────────┘
```

#### 4.2.2 질의-응답 파이프라인

```
┌─────────────┐
│ 사용자 질문  │
└──────┬──────┘
       ↓
┌─────────────────────────────────────────┐
│ Router (agent/router.py)                │
│ - 질문 유형 분류 (LLM)                   │
│   • 문서검색: RAG 필요                   │
│   • 직접질의: 일반 상식, 계산            │
│   • 복합질문: 다단계 처리                │
│   • 불확실: 안전 응답                    │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ [문서검색 경로]                          │
│                                          │
│ Retriever (rag/retriever.py)            │
│ 1. 질문 임베딩 생성                      │
│ 2. 하이브리드 검색                       │
│    - 벡터 검색 (코사인 유사도)           │
│    - BM25 검색 (키워드)                  │
│    - 메타데이터 필터 (부서, 날짜)        │
│ 3. 결과 병합 (가중치: 0.7:0.3)           │
│ 4. Top-K 선택 (K=20)                     │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Re-ranker (rag/retriever.py)            │
│ - flashrank 모델                         │
│ - 질문-청크 관련성 재평가                │
│ - Top-N 최종 선택 (N=5)                  │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ Prompt Builder (rag/prompt.py)          │
│ - 시스템 프롬프트 로드                   │
│ - Few-shot 예시 삽입                     │
│ - 검색된 청크 컨텍스트 추가              │
│ - 질문 삽입                              │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ LLM (core/llm/)                         │
│ - 스트리밍 생성                          │
│ - 출처 표시 지시                         │
│ - 환각 방지 프롬프트                     │
└──────┬──────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ 응답 + 출처                              │
│ - 답변 텍스트                            │
│ - 참조 청크 리스트                       │
│ - 신뢰도 점수                            │
└─────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────┐
│ [직접질의 경로] (SFR-011)                │
│                                          │
│ - 지식 참조 없이 LLM 직접 호출           │
│ - 일반 상식, 계산, 번역 등               │
│ - 출처 없음                              │
└─────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────┐
│ [복합질문 경로]                          │
│                                          │
│ Planner (agent/planner.py)              │
│ 1. 질문 분해 (LLM)                       │
│    "A와 B의 차이는?" → [A 정의, B 정의, 비교]│
│ 2. 서브쿼리 순차 실행                    │
│ 3. 결과 종합                             │
│ 4. 최종 응답 생성                        │
└─────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────┐
│ [불확실 경로]                            │
│                                          │
│ - 표준 안내 문구 반환                    │
│ - "죄송합니다. 해당 질문은..."           │
│ - 안전 장치                              │
└─────────────────────────────────────────┘
```

### 4.3 컴포넌트 간 통신

#### 4.3.1 Frontend ↔ Backend

- **프로토콜**: HTTP/HTTPS, Server-Sent Events (SSE)
- **포맷**: JSON
- **인증**: JWT (시연용 간소화)
- **스트리밍**: SSE로 실시간 응답 전송

#### 4.3.2 Backend 내부

- **계층 분리**: API → Service → Core
- **의존성 주입**: FastAPI Depends
- **비동기**: asyncio 기반

#### 4.3.3 외부 서비스

- **Upstage OCR API**: HTTPS REST
- **OpenAI API**: HTTPS REST (선택적)
- **Anthropic API**: HTTPS REST (선택적)

---

## 5. 기능 요구사항 매핑

### 5.1 SFR (시스템 기능 요구사항) 상세 매핑

| 요구사항 ID | 요구사항 명 | 구현 컴포넌트 | 기술 스택 | 우선순위 | 구현 상세 |
|------------|-----------|-------------|----------|---------|----------|
| **SFR-001** | 일반사항 | `frontend/` | React, TypeScript, MUI | P1 | - 반응형 디자인 (모바일/태블릿/데스크톱)<br>- 시큐어 코딩 (XSS/CSRF 방지)<br>- 크로스 브라우저 (Chrome, Edge, Safari)<br>- 접근성 (WCAG 2.1 AA) |
| **SFR-002** | 접근관리 | `api/routes/auth.py`<br>`api/middleware.py` | FastAPI, JWT | P2 | - SSO/EAM 연동 (시연에서는 mock)<br>- 역할 기반 접근 제어 (RBAC)<br>- 세션 관리<br>- 감사 로그 |
| **SFR-003** | 모델관리 | `api/routes/admin.py`<br>`core/llm/` | FastAPI, SQLite | P1 | - 모델 CRUD (추가/수정/삭제)<br>- 프롬프트 버전 관리<br>- 파라미터 설정 (temperature, max_tokens)<br>- 유해 컨텐츠 필터링 설정 |
| **SFR-004** | LLM 기본기능 | `core/llm/`<br>`rag/chain.py` | Ollama, OpenAI | P0 | - 스트리밍 응답 (SSE)<br>- 멀티턴 대화 (메모리)<br>- 환각 방지 (출처 기반 응답)<br>- 다국어 (한/영) |
| **SFR-005** | RAG 지식관리 | `pipeline/`<br>`api/routes/documents.py` | ChromaDB, SQLite | P0 | - 폴더별 권한 관리<br>- 배치 동기화<br>- 개인 영역 (user_id 필터)<br>- PII 감지 (정규식 기반) |
| **SFR-006** | 생성형AI 플랫폼 | `frontend/ChatPage.tsx`<br>`api/routes/chat.py` | React, FastAPI | P0 | - 채팅 UI (말풍선, 타이핑 효과)<br>- 출처 표시 (클릭 시 원문)<br>- 다양한 포맷 출력 (표, 코드)<br>- 요약/번역 기능<br>- 스트리밍 |
| **SFR-007** | RAG 파이프라인 | `pipeline/`<br>`rag/retriever.py` | PyMuPDF, ChromaDB | P0 | - 문서 수집 (다중 포맷)<br>- 전처리 (청킹, OCR)<br>- 임베딩 생성<br>- 벡터 저장<br>- 하이브리드 검색<br>- Re-ranking<br>- 출처 추적 |
| **SFR-008** | RAG 품질관리 | `frontend/MonitorPage.tsx`<br>`api/routes/monitoring.py` | React, SQLite | P1 | - 청크 품질 모니터링 (길이, 중복)<br>- 임베딩 품질 대시보드<br>- 재처리 큐<br>- 오류 로그 |
| **SFR-009** | 정보서비스 관리 | `api/routes/admin.py`<br>`frontend/AdminPage.tsx` | FastAPI, React | P2 | - 사용 통계 (일/주/월)<br>- 리포트 생성<br>- 콘텐츠 관리 (FAQ, 공지)<br>- 사용자 관리 |
| **SFR-010** | 로그/통계 | `api/middleware.py`<br>`api/routes/monitoring.py` | SQLite, FastAPI | P1 | - 질의 로그 (질문, 응답, 시간)<br>- 접속 통계 (사용자, IP, 시간)<br>- APM (응답 시간, 처리량)<br>- 사용량 대시보드 |
| **SFR-011** | 다중응답모드 | `frontend/ChatPage.tsx`<br>`api/routes/chat.py` | React, FastAPI | P1 | - UI 토글 (직접모드/지식참조모드)<br>- 라우터 분기 처리<br>- 모드별 프롬프트 |
| **SFR-012** | 범용 RAG | `pipeline/loader.py`<br>`pipeline/chunker.py` | PyMuPDF, ChromaDB | P0 | - 다중 포맷 지원 (PDF, HWP, XLSX 등)<br>- 표 추출 (구조 보존)<br>- 시맨틱 청킹<br>- HNSW 벡터 인덱스 |
| **SFR-013** | 에이전트 서비스 | `agent/` | LangChain, MCP | P1 | - 시범 에이전트 (문서검색, 계산기)<br>- 멀티에이전트 협업<br>- Tool 호출 |
| **SFR-014** | 관리자 포털 | `frontend/AdminPage.tsx`<br>`api/routes/admin.py` | React, FastAPI | P1 | - 모델 설정 UI<br>- 사용자 관리<br>- 지식 영역 관리<br>- 문서 관리<br>- 통계 대시보드<br>- SW 모니터링 |
| **SFR-015** | OCR | `pipeline/ocr.py` | Upstage API | P1 | - 스캔 PDF 감지<br>- Upstage API 호출<br>- 신뢰도 점수<br>- 문맥 보정 (LLM) |
| **SFR-016** | Re-ranking | `rag/retriever.py` | flashrank | P0 | - 하이브리드 검색 (BM25 + 벡터)<br>- Re-ranking 파이프라인<br>- 설정 가능 가중치<br>- 모델 교체 가능 |
| **SFR-017** | 답변품질향상 | `rag/quality.py`<br>`api/routes/feedback.py` | SQLite, FastAPI | P0 | - 전문가 리뷰 시스템<br>- Golden Data 관리<br>- 신뢰도 임계값 설정<br>- 자동 품질 평가 |
| **SFR-018** | 에이전트 플랫폼 | `agent/mcp/`<br>`agent/tool_registry.py`<br>`agent/builder.py` | MCP, FastAPI | P0 (API), P2 (UI) | - LLM-hub 개념 (멀티 모델) - P0<br>- MCP 서버 구현 - P0<br>- Tool 등록 시스템 - P0<br>- Agent 워크플로우 실행 - P0<br>- Agent Builder 비주얼 캔버스 UI - P2 (향후 구현) |

### 5.2 ECR (환경 요구사항) 매핑

| 요구사항 ID | 요구사항 명 | 구현 방식 | 우선순위 |
|------------|-----------|----------|---------|
| **ECR-001** | 서버 환경 | Ubuntu Linux, Docker | P0 |
| **ECR-002** | GPU 환경 | NVIDIA H200, L40S, CUDA 12+ | P0 |
| **ECR-003** | 네트워크 환경 | 망분리, 내부망 전용 | P0 |
| **ECR-004** | 스토리지 환경 | NVMe SSD, 영구 볼륨 | P0 |

### 5.3 INR (인터페이스 요구사항) 매핑

| 요구사항 ID | 요구사항 명 | 구현 컴포넌트 | 우선순위 |
|------------|-----------|-------------|---------|
| **INR-001** | REST API | `api/main.py` | P0 |
| **INR-002** | 스트리밍 API | SSE 엔드포인트 | P0 |
| **INR-003** | 웹 UI | `frontend/` | P0 |
| **INR-004** | 관리자 UI | `frontend/AdminPage.tsx` | P1 |
| **INR-005** | MCP 인터페이스 | `agent/mcp/server.py` | P1 |
| **INR-006** | 외부 API 연동 | Upstage OCR, OpenAI (선택) | P1 |
| **INR-007** | 데이터베이스 연동 | SQLite, ChromaDB | P0 |

### 5.4 DAR (데이터 요구사항) 매핑

| 요구사항 ID | 요구사항 명 | 구현 방식 | 우선순위 |
|------------|-----------|----------|---------|
| **DAR-001** | 문서 저장 | 파일 시스템 + 메타데이터 DB | P0 |
| **DAR-002** | 벡터 저장 | ChromaDB (HNSW 인덱스) | P0 |
| **DAR-003** | 데이터 규모 | ~2,000 문서, ~35GB | P0 |
| **DAR-004** | 메타데이터 스키마 | SQLite (documents, chunks 테이블) | P0 |
| **DAR-005** | 백업/복구 | 볼륨 스냅샷 | P2 |
| **DAR-006** | 데이터 보안 | PII 감지, 접근 제어 | P1 |

### 5.5 기타 요구사항 매핑

#### TER (기술 요구사항)
- **TER-001**: Python 3.11+, React 18+, FastAPI, ChromaDB → **P0**

#### SER (시스템 환경 요구사항)
- **SER-001 ~ SER-007**: Docker, Kubernetes (선택), 모니터링, 로깅 → **P1-P2**

#### QUR (품질 요구사항)
- **QUR-001**: 응답 시간 ≤ 3초 → **P0**
- **QUR-002**: 동시 사용자 ≥ 50명 → **P1**
- **QUR-003**: 가용성 ≥ 99% (운영 시) → **P2**
- **QUR-004**: RAG 정확도 ≥ 90% → **P0**

#### COR (제약 요구사항)
- **COR-001**: 망분리 환경, 1개월 기한 → **P0**

#### PMR (프로젝트 관리 요구사항)
- **PMR-001**: 문서화, 테스트, 배포 가이드 → **P1**

#### PSR (프로세스 요구사항)
- **PSR-001**: 지속적 통합/배포 (CI/CD) → **P2**
- **PSR-002**: 코드 리뷰, 테스트 자동화 → **P2**

---

## 6. 데이터 모델

### 6.1 관계형 데이터베이스 (SQLite)

#### 6.1.1 documents 테이블

문서 메타데이터 저장

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,              -- UUID
    filename TEXT NOT NULL,            -- 원본 파일명
    file_type TEXT NOT NULL,           -- pdf, hwp, xlsx, docx, pptx, txt
    file_size INTEGER NOT NULL,        -- 바이트 단위
    file_path TEXT NOT NULL,           -- 저장 경로
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    department TEXT,                   -- 부서명 (메타데이터)
    category TEXT,                     -- 문서 유형 (보고서, 공문 등)
    status TEXT DEFAULT 'pending',     -- pending, processing, completed, failed
    chunk_count INTEGER DEFAULT 0,     -- 생성된 청크 수
    user_id TEXT,                      -- 업로드 사용자
    is_public BOOLEAN DEFAULT 1,       -- 공개/비공개
    ocr_applied BOOLEAN DEFAULT 0,     -- OCR 적용 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);
```

#### 6.1.2 chunks 테이블

문서 청크 정보 (벡터는 ChromaDB에 저장)

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,              -- UUID
    document_id TEXT NOT NULL,         -- documents.id 참조
    content TEXT NOT NULL,             -- 청크 텍스트
    chunk_index INTEGER NOT NULL,      -- 문서 내 순서
    metadata_json TEXT,                -- JSON 형태 메타데이터
    embedding_id TEXT,                 -- ChromaDB 벡터 ID (동일)
    token_count INTEGER,               -- 토큰 수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding_id ON chunks(embedding_id);
```

#### 6.1.3 conversations 테이블

대화 세션

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,              -- UUID
    session_id TEXT NOT NULL UNIQUE,   -- 세션 식별자
    user_id TEXT NOT NULL,             -- 사용자 ID
    title TEXT,                        -- 대화 제목 (첫 질문)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
```

#### 6.1.4 messages 테이블

대화 메시지

```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,              -- UUID
    conversation_id TEXT NOT NULL,     -- conversations.id 참조
    role TEXT NOT NULL,                -- user, assistant, system
    content TEXT NOT NULL,             -- 메시지 내용
    sources_json TEXT,                 -- JSON 배열 (참조 청크들)
    confidence_score REAL,             -- 신뢰도 점수 (0.0-1.0)
    response_mode TEXT,                -- direct, rag, hybrid
    model_used TEXT,                   -- 사용된 모델명
    latency_ms INTEGER,                -- 응답 시간 (밀리초)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

#### 6.1.5 feedback 테이블

사용자 피드백

```sql
CREATE TABLE feedback (
    id TEXT PRIMARY KEY,              -- UUID
    message_id TEXT NOT NULL,          -- messages.id 참조
    user_id TEXT NOT NULL,
    rating TEXT CHECK(rating IN ('up', 'down')),
    correction_text TEXT,              -- 사용자가 제공한 정답
    reason TEXT,                       -- 피드백 이유
    is_reviewed BOOLEAN DEFAULT 0,     -- 전문가 리뷰 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX idx_feedback_message_id ON feedback(message_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
```

#### 6.1.6 golden_data 테이블

Golden Q&A 데이터셋

```sql
CREATE TABLE golden_data (
    id TEXT PRIMARY KEY,              -- UUID
    question TEXT NOT NULL,            -- 질문
    answer TEXT NOT NULL,              -- 정답
    sources_json TEXT,                 -- JSON 배열 (출처 청크)
    category TEXT,                     -- 질문 유형
    created_by TEXT,                   -- 생성자 (전문가)
    is_active BOOLEAN DEFAULT 1,       -- 활성화 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_golden_data_category ON golden_data(category);
CREATE INDEX idx_golden_data_is_active ON golden_data(is_active);
```

#### 6.1.7 prompts 테이블

프롬프트 버전 관리

```sql
CREATE TABLE prompts (
    id TEXT PRIMARY KEY,              -- UUID
    name TEXT NOT NULL,                -- 프롬프트 이름
    version INTEGER NOT NULL,          -- 버전 번호
    system_prompt TEXT NOT NULL,       -- 시스템 프롬프트
    few_shot_examples_json TEXT,       -- JSON 배열 (Few-shot 예시)
    is_active BOOLEAN DEFAULT 0,       -- 현재 활성 버전
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE INDEX idx_prompts_name ON prompts(name);
CREATE INDEX idx_prompts_is_active ON prompts(is_active);
```

#### 6.1.8 models 테이블

LLM 모델 설정

```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,              -- UUID
    name TEXT NOT NULL UNIQUE,         -- 모델 표시명
    provider TEXT NOT NULL,            -- ollama, openai, anthropic
    model_id TEXT NOT NULL,            -- 실제 모델 ID (예: llama3)
    config_json TEXT,                  -- JSON 형태 설정 (temperature 등)
    is_active BOOLEAN DEFAULT 1,       -- 사용 가능 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_models_provider ON models(provider);
CREATE INDEX idx_models_is_active ON models(is_active);
```

#### 6.1.9 agents 테이블

에이전트 정의

```sql
CREATE TABLE agents (
    id TEXT PRIMARY KEY,              -- UUID
    name TEXT NOT NULL UNIQUE,         -- 에이전트 이름
    description TEXT,
    workflow_json TEXT NOT NULL,       -- JSON 워크플로우 정의
    tools_json TEXT,                   -- JSON 배열 (사용 도구들)
    model_id TEXT,                     -- 기본 모델
    is_active BOOLEAN DEFAULT 1,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agents_is_active ON agents(is_active);
```

#### 6.1.10 tools 테이블

도구 레지스트리

```sql
CREATE TABLE tools (
    id TEXT PRIMARY KEY,              -- UUID
    name TEXT NOT NULL UNIQUE,         -- 도구 이름
    description TEXT,
    tool_type TEXT NOT NULL CHECK(tool_type IN ('mcp', 'api', 'function')),
    config_json TEXT NOT NULL,         -- JSON 형태 설정
    is_active BOOLEAN DEFAULT 1,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tools_tool_type ON tools(tool_type);
CREATE INDEX idx_tools_is_active ON tools(is_active);
```

#### 6.1.11 logs 테이블

시스템 로그

```sql
CREATE TABLE logs (
    id TEXT PRIMARY KEY,              -- UUID
    user_id TEXT,
    action TEXT NOT NULL,              -- 액션 유형
    detail_json TEXT,                  -- JSON 상세 정보
    ip_address TEXT,
    user_agent TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_logs_user_id ON logs(user_id);
CREATE INDEX idx_logs_action ON logs(action);
CREATE INDEX idx_logs_created_at ON logs(created_at);
```

### 6.2 벡터 데이터베이스 (ChromaDB)

#### 6.2.1 knowledge_base 컬렉션

문서 청크 임베딩

```python
collection = chroma_client.create_collection(
    name="knowledge_base",
    metadata={
        "description": "Main document chunks",
        "embedding_model": "bge-m3",
        "dimension": 1024
    },
    embedding_function=embeddings
)

# 각 벡터의 메타데이터 스키마
metadata_schema = {
    "chunk_id": str,           # chunks.id와 동일
    "document_id": str,        # 문서 ID
    "filename": str,           # 원본 파일명
    "chunk_index": int,        # 청크 순서
    "department": str,         # 부서명
    "category": str,           # 문서 유형
    "date": str,               # 문서 날짜 (ISO 8601)
    "page_number": int,        # 페이지 번호 (PDF)
    "user_id": str,            # 소유자 (개인 영역)
    "is_public": bool          # 공개 여부
}
```

#### 6.2.2 golden_data 컬렉션

Golden Q&A 임베딩

```python
collection = chroma_client.create_collection(
    name="golden_data",
    metadata={
        "description": "Curated Q&A pairs",
        "embedding_model": "bge-m3",
        "dimension": 1024
    },
    embedding_function=embeddings
)

# 메타데이터 스키마
metadata_schema = {
    "golden_id": str,          # golden_data.id와 동일
    "question": str,           # 질문
    "answer": str,             # 정답
    "category": str,           # 질문 유형
    "created_by": str          # 전문가
}
```

---

## 7. API 설계

### 7.1 API 개요

- **Base URL**: `http://localhost:8000/api`
- **인증**: JWT Bearer Token (헤더: `Authorization: Bearer <token>`)
- **응답 형식**: JSON
- **에러 형식**: RFC 7807 Problem Details
- **버전**: v1 (URL에 명시 안 함, 향후 v2는 `/api/v2/...`)

### 7.2 공통 응답 스키마

#### 성공 응답
```json
{
    "success": true,
    "data": { /* 응답 데이터 */ },
    "message": "Success"
}
```

#### 에러 응답
```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": { /* 상세 정보 */ }
    }
}
```

### 7.3 Chat API (`/api/chat`)

#### POST /api/chat
채팅 메시지 전송 (SSE 스트리밍)

**Request:**
```json
{
    "message": "한국가스기술공사의 주요 사업은?",
    "session_id": "uuid-string",      // 선택적, 없으면 신규 생성
    "mode": "rag",                     // rag | direct | hybrid
    "model": "gpt-4",                  // 선택적, 기본값 사용
    "filters": {                       // 선택적 메타데이터 필터
        "department": "기술연구소",
        "date_from": "2023-01-01",
        "date_to": "2024-12-31"
    }
}
```

**Response (SSE Stream):**
```
event: start
data: {"message_id": "uuid", "session_id": "uuid"}

event: chunk
data: {"content": "한국가스기술공사의"}

event: chunk
data: {"content": " 주요 사업은"}

event: source
data: {"chunk_id": "uuid", "filename": "사업보고서.pdf", "page": 5}

event: end
data: {"confidence_score": 0.92, "latency_ms": 2341}
```

#### POST /api/chat/cancel
현재 생성 중인 응답 취소

**Request:**
```json
{
    "message_id": "uuid-string"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Generation cancelled"
}
```

#### GET /api/chat/sessions
사용자의 세션 목록 조회

**Query Parameters:**
- `limit`: 페이지 크기 (기본: 20)
- `offset`: 오프셋 (기본: 0)

**Response:**
```json
{
    "success": true,
    "data": {
        "sessions": [
            {
                "id": "uuid",
                "session_id": "uuid",
                "title": "한국가스기술공사 사업",
                "created_at": "2026-02-05T10:00:00Z",
                "updated_at": "2026-02-05T11:30:00Z",
                "message_count": 12
            }
        ],
        "total": 45,
        "limit": 20,
        "offset": 0
    }
}
```

#### GET /api/chat/sessions/{session_id}/messages
세션의 메시지 목록

**Response:**
```json
{
    "success": true,
    "data": {
        "messages": [
            {
                "id": "uuid",
                "role": "user",
                "content": "질문 내용",
                "created_at": "2026-02-05T10:00:00Z"
            },
            {
                "id": "uuid",
                "role": "assistant",
                "content": "답변 내용",
                "sources": [
                    {
                        "chunk_id": "uuid",
                        "filename": "보고서.pdf",
                        "page": 5,
                        "content": "출처 텍스트..."
                    }
                ],
                "confidence_score": 0.92,
                "response_mode": "rag",
                "latency_ms": 2341,
                "created_at": "2026-02-05T10:00:05Z"
            }
        ]
    }
}
```

#### DELETE /api/chat/sessions/{session_id}
세션 삭제

**Response:**
```json
{
    "success": true,
    "message": "Session deleted"
}
```

### 7.4 Documents API (`/api/documents`)

#### POST /api/documents/upload
문서 업로드 (다중 파일 지원)

**Request (multipart/form-data):**
```
files: [File, File, ...]
department: "기술연구소"           // 선택적
category: "기술보고서"             // 선택적
is_public: true                    // 선택적 (기본: true)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "uploaded": [
            {
                "id": "uuid",
                "filename": "보고서.pdf",
                "file_size": 1048576,
                "status": "processing"
            }
        ],
        "failed": []
    }
}
```

#### GET /api/documents
문서 목록 조회

**Query Parameters:**
- `limit`: 페이지 크기 (기본: 20)
- `offset`: 오프셋 (기본: 0)
- `category`: 카테고리 필터
- `department`: 부서 필터
- `status`: 상태 필터 (pending, processing, completed, failed)
- `search`: 파일명 검색

**Response:**
```json
{
    "success": true,
    "data": {
        "documents": [
            {
                "id": "uuid",
                "filename": "보고서.pdf",
                "file_type": "pdf",
                "file_size": 1048576,
                "upload_date": "2026-02-05T09:00:00Z",
                "department": "기술연구소",
                "category": "기술보고서",
                "status": "completed",
                "chunk_count": 45,
                "is_public": true,
                "ocr_applied": false
            }
        ],
        "total": 2000,
        "limit": 20,
        "offset": 0
    }
}
```

#### GET /api/documents/{document_id}
문서 상세 조회

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "filename": "보고서.pdf",
        "file_type": "pdf",
        "file_size": 1048576,
        "file_path": "/data/documents/uuid.pdf",
        "upload_date": "2026-02-05T09:00:00Z",
        "department": "기술연구소",
        "category": "기술보고서",
        "status": "completed",
        "chunk_count": 45,
        "user_id": "user-uuid",
        "is_public": true,
        "ocr_applied": false,
        "created_at": "2026-02-05T09:00:00Z",
        "updated_at": "2026-02-05T09:05:00Z"
    }
}
```

#### DELETE /api/documents/{document_id}
문서 삭제 (청크 및 벡터도 함께 삭제)

**Response:**
```json
{
    "success": true,
    "message": "Document deleted"
}
```

#### POST /api/documents/{document_id}/reprocess
문서 재처리 (청킹, 임베딩 재생성)

**Request:**
```json
{
    "apply_ocr": true,                 // 선택적, OCR 재적용
    "chunk_size": 800                  // 선택적, 청크 크기 변경
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "status": "processing",
        "message": "Document reprocessing started"
    }
}
```

#### GET /api/documents/{document_id}/chunks
문서의 청크 목록

**Query Parameters:**
- `limit`: 페이지 크기 (기본: 50)
- `offset`: 오프셋 (기본: 0)

**Response:**
```json
{
    "success": true,
    "data": {
        "chunks": [
            {
                "id": "uuid",
                "content": "청크 텍스트 내용...",
                "chunk_index": 0,
                "metadata": {
                    "page_number": 1,
                    "department": "기술연구소"
                },
                "token_count": 512
            }
        ],
        "total": 45,
        "limit": 50,
        "offset": 0
    }
}
```

### 7.5 Admin API (`/api/admin`)

#### GET /api/admin/models
모델 목록 조회

**Response:**
```json
{
    "success": true,
    "data": {
        "models": [
            {
                "id": "uuid",
                "name": "GPT-4",
                "provider": "openai",
                "model_id": "gpt-4",
                "config": {
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "is_active": true
            }
        ]
    }
}
```

#### PUT /api/admin/models/{model_id}
모델 설정 수정

**Request:**
```json
{
    "name": "GPT-4 Updated",
    "config": {
        "temperature": 0.5,
        "max_tokens": 4096
    },
    "is_active": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "name": "GPT-4 Updated",
        "updated_at": "2026-02-05T12:00:00Z"
    }
}
```

#### GET /api/admin/prompts
프롬프트 목록

**Response:**
```json
{
    "success": true,
    "data": {
        "prompts": [
            {
                "id": "uuid",
                "name": "default",
                "version": 3,
                "is_active": true,
                "created_at": "2026-02-01T00:00:00Z"
            }
        ]
    }
}
```

#### PUT /api/admin/prompts/{prompt_id}
프롬프트 수정 (새 버전 생성)

**Request:**
```json
{
    "system_prompt": "새 시스템 프롬프트",
    "few_shot_examples": [
        {
            "question": "예시 질문",
            "answer": "예시 답변"
        }
    ],
    "is_active": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "new-uuid",
        "name": "default",
        "version": 4,
        "is_active": true
    }
}
```

#### GET /api/admin/prompts/{name}/versions
프롬프트 버전 이력

**Response:**
```json
{
    "success": true,
    "data": {
        "versions": [
            {
                "version": 4,
                "is_active": true,
                "created_at": "2026-02-05T12:00:00Z"
            },
            {
                "version": 3,
                "is_active": false,
                "created_at": "2026-02-01T00:00:00Z"
            }
        ]
    }
}
```

#### POST /api/admin/prompts/{prompt_id}/rollback
이전 버전으로 롤백

**Response:**
```json
{
    "success": true,
    "message": "Rolled back to version 3"
}
```

### 7.6 Feedback API (`/api/feedback`)

#### POST /api/feedback
피드백 제출

**Request:**
```json
{
    "message_id": "uuid",
    "rating": "down",                  // up | down
    "correction_text": "정확한 답변은...",  // 선택적
    "reason": "부정확한 출처"            // 선택적
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "created_at": "2026-02-05T13:00:00Z"
    }
}
```

#### GET /api/feedback
피드백 목록 (관리자)

**Query Parameters:**
- `rating`: up | down | all (기본: all)
- `is_reviewed`: true | false
- `limit`, `offset`

**Response:**
```json
{
    "success": true,
    "data": {
        "feedback": [
            {
                "id": "uuid",
                "message_id": "uuid",
                "user_id": "user-uuid",
                "rating": "down",
                "correction_text": "정확한 답변은...",
                "is_reviewed": false,
                "created_at": "2026-02-05T13:00:00Z"
            }
        ],
        "total": 23
    }
}
```

#### POST /api/feedback/{feedback_id}/golden
피드백을 Golden Data로 승격

**Request:**
```json
{
    "category": "기술문의",
    "approved_by": "expert-user-uuid"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "golden_id": "uuid",
        "message": "Promoted to golden data"
    }
}
```

### 7.7 Agent API (`/api/agents`)

#### GET /api/agents
에이전트 목록

**Response:**
```json
{
    "success": true,
    "data": {
        "agents": [
            {
                "id": "uuid",
                "name": "문서검색 에이전트",
                "description": "문서에서 정보를 찾습니다",
                "is_active": true,
                "tools": ["search", "summarize"]
            }
        ]
    }
}
```

#### POST /api/agents
에이전트 생성

**Request:**
```json
{
    "name": "계산기 에이전트",
    "description": "수학 계산을 수행합니다",
    "workflow": {
        "steps": [
            {
                "type": "tool_call",
                "tool": "calculator",
                "params": {}
            }
        ]
    },
    "tools": ["calculator"],
    "model_id": "gpt-4"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "name": "계산기 에이전트",
        "created_at": "2026-02-05T14:00:00Z"
    }
}
```

#### PUT /api/agents/{agent_id}
에이전트 수정

**Request:**
```json
{
    "description": "수정된 설명",
    "workflow": { /* 업데이트된 워크플로우 */ }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "updated_at": "2026-02-05T14:30:00Z"
    }
}
```

#### POST /api/agents/{agent_id}/execute
에이전트 실행

**Request:**
```json
{
    "input": "2의 10승은?",
    "context": {}                      // 선택적
}
```

**Response (SSE Stream):**
```
event: start
data: {"execution_id": "uuid"}

event: step
data: {"step": 1, "tool": "calculator", "status": "running"}

event: result
data: {"step": 1, "output": "1024"}

event: end
data: {"final_output": "2의 10승은 1024입니다.", "latency_ms": 543}
```

#### GET /api/agents/tools
도구 목록

**Response:**
```json
{
    "success": true,
    "data": {
        "tools": [
            {
                "id": "uuid",
                "name": "calculator",
                "description": "수학 계산",
                "tool_type": "function",
                "is_active": true
            },
            {
                "id": "uuid",
                "name": "search",
                "description": "문서 검색",
                "tool_type": "mcp",
                "is_active": true
            }
        ]
    }
}
```

#### POST /api/agents/tools
도구 등록

**Request:**
```json
{
    "name": "weather",
    "description": "날씨 정보 조회",
    "tool_type": "api",
    "config": {
        "url": "https://api.weather.com",
        "auth_type": "api_key"
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "uuid",
        "name": "weather",
        "created_at": "2026-02-05T15:00:00Z"
    }
}
```

#### GET /api/agents/mcp/status
MCP 서버 상태

**Response:**
```json
{
    "success": true,
    "data": {
        "status": "running",
        "uptime_seconds": 3600,
        "connected_tools": 5,
        "version": "1.0.0"
    }
}
```

### 7.8 Monitoring API (`/api/monitoring`)

#### GET /api/monitoring/stats
사용 통계

**Query Parameters:**
- `period`: day | week | month (기본: day)
- `from`: 시작 날짜 (ISO 8601)
- `to`: 종료 날짜 (ISO 8601)

**Response:**
```json
{
    "success": true,
    "data": {
        "total_queries": 1234,
        "unique_users": 45,
        "avg_response_time_ms": 2341,
        "success_rate": 0.95,
        "top_categories": [
            {"category": "기술문의", "count": 345},
            {"category": "일반문의", "count": 234}
        ]
    }
}
```

#### GET /api/monitoring/logs
로그 조회

**Query Parameters:**
- `action`: 액션 필터
- `user_id`: 사용자 필터
- `from`, `to`: 날짜 범위
- `limit`, `offset`

**Response:**
```json
{
    "success": true,
    "data": {
        "logs": [
            {
                "id": "uuid",
                "user_id": "user-uuid",
                "action": "chat_query",
                "detail": {
                    "message": "질문 내용",
                    "mode": "rag"
                },
                "ip_address": "192.168.1.100",
                "response_time_ms": 2341,
                "created_at": "2026-02-05T16:00:00Z"
            }
        ],
        "total": 5678
    }
}
```

#### GET /api/monitoring/rag-quality
RAG 품질 지표

**Response:**
```json
{
    "success": true,
    "data": {
        "recall": 0.92,
        "precision": 0.91,
        "semantic_similarity": 0.93,
        "avg_confidence_score": 0.88,
        "evaluated_pairs": 156,
        "last_updated": "2026-02-05T16:00:00Z"
    }
}
```

#### GET /api/monitoring/embeddings
임베딩 현황

**Response:**
```json
{
    "success": true,
    "data": {
        "total_embeddings": 98765,
        "embedding_dimension": 1024,
        "model": "bge-m3",
        "collections": [
            {
                "name": "knowledge_base",
                "count": 98000,
                "size_mb": 384
            },
            {
                "name": "golden_data",
                "count": 765,
                "size_mb": 3
            }
        ]
    }
}
```

---

## 8. UI/UX 설계

### 8.1 ChatPage (메인 챗봇)

**레이아웃 구조**
```
┌─────────────┬────────────────────────────────────────────┐
│   사이드바   │              채팅 영역                      │
│             │  ┌──────────────────────────────────────┐  │
│ 📁 세션목록 │  │  상단: 모델선택 | 응답모드 토글       │  │
│  - 오늘     │  ├──────────────────────────────────────┤  │
│  - 어제     │  │                                      │  │
│  - 지난주   │  │      채팅 메시지 영역                 │  │
│             │  │      (스크롤 가능)                    │  │
│ [+ 새채팅]  │  │                                      │  │
│             │  └──────────────────────────────────────┘  │
│             │  [파일첨부] [입력창...] [전송]            │
└─────────────┴────────────────────────────────────────────┘
```

**좌측 사이드바 (240px 고정폭)**
- 세션 목록: 시간순 그룹핑 (오늘/어제/지난 7일/이전)
- 각 세션: 첫 질문 텍스트 표시 (최대 40자), 클릭시 해당 채팅 로드
- 새 채팅 버튼: 상단 고정, 항상 접근 가능
- 세션 호버시: 이름변경/삭제/공유 아이콘 표시
- 폴더 기능: 세션을 주제별로 그룹화 (선택사항)

**상단 컨트롤 바**
- 모델 선택 드롭다운:
  - Ollama: qwen2.5:7b, llama3.1:8b 등
  - OpenAI: gpt-4o, gpt-4o-mini
  - Claude: claude-3-5-sonnet-20241022
  - 모델 전환시 즉시 반영, 기존 대화는 유지
- 응답모드 토글 (SFR-011):
  - 🔍 지식참조 (RAG): 벡터DB 검색 후 답변
  - 💬 직접응답: LLM만 사용, 빠른 대화형
  - 토글 상태 세션별 저장
- 설정 아이콘: 온도, max_tokens, 스트리밍 설정

**중앙 채팅 영역**
- 메시지 카드 레이아웃:
  ```
  ┌────────────────────────────────────────┐
  │ 👤 사용자                               │
  │ 올해 예산 편성 기준이 뭐야?              │
  └────────────────────────────────────────┘

  ┌────────────────────────────────────────┐
  │ 🤖 AI Assistant        [신뢰도: 높음 🟢] │
  │                                        │
  │ 2024년 예산 편성 기준은 다음과 같습니다: │
  │ 1. 전년도 대비 5% 증액 원칙...          │
  │                                        │
  │ 📎 출처 문서:                           │
  │ • 2024_예산편성지침.pdf (p.3, 신뢰도 0.92) │
  │ • 재정운영규칙.hwp (1장, 신뢰도 0.87)    │
  │                                        │
  │ [👍 좋아요] [👎 싫어요] [📝 보정하기]    │
  └────────────────────────────────────────┘
  ```

**신뢰도 배지 (SFR-012)**
- 🟢 높음 (0.8~1.0): 녹색 배지, 검색된 문서와 높은 유사도
- 🟡 중간 (0.5~0.8): 황색 배지, 부분적 일치
- 🔴 낮음 (0.0~0.5): 적색 배지, "확실하지 않은 답변입니다" 경고 표시

**출처 문서 리스트**
- 클릭시 모달로 원문 청크 표시
- 하이라이트: 검색 쿼리와 매칭된 부분 강조
- 메타데이터: 페이지번호, 섹션, 작성자, 작성일
- 다운로드/공유 버튼

**피드백 메커니즘**
- 👍 좋아요: DB에 긍정 피드백 저장, Golden Data 후보로 마킹
- 👎 싫어요: 부정 피드백 + 보정 텍스트 입력 모달 오픈
- 📝 보정하기: "올바른 답변은 이것입니다" 입력 → 운영자 검토 큐로 이동
- 피드백 즉시 반영: Golden Data 승인시 벡터DB 추가

**라우팅 시각화 (SFR-018)**
- 작은 패널 (우측 상단, 접기 가능):
  ```
  🔄 질문 분류 중...
  ✅ 카테고리: 문서검색
  🛠️ 실행 계획: RAG 검색 → 답변 생성
  📊 검색된 청크: 3개
  ```
- 개발자/관리자 모드에서만 표시 (일반 사용자는 숨김 옵션)

**하단 입력 영역**
- 텍스트 입력창: 1000자 이상 지원, 자동 높이 조절 (최대 5줄)
- 파일 첨부: 드래그&드롭 또는 클릭 업로드
  - 지원 형식: PDF, HWP, DOCX, XLSX, TXT, 이미지
  - 첨부 파일은 임시 업로드 → OCR/파싱 → 컨텍스트에 포함
- 전송 버튼: 입력 있을 때만 활성화
- 취소 버튼: 스트리밍 응답 중단

**스트리밍 응답**
- SSE(Server-Sent Events)로 실시간 텍스트 수신
- 마크다운 실시간 렌더링 (코드블록, 테이블, 리스트 등)
- 응답 중 로딩 인디케이터: "생각하는 중..." 애니메이션

**키보드 단축키**
- `Cmd+Enter` (Mac) / `Ctrl+Enter` (Win): 전송
- `Cmd+K`: 새 채팅
- `Esc`: 스트리밍 중단

---

### 8.2 DocumentsPage (문서 관리)

**레이아웃 구조**
```
┌──────────────┬─────────────────────────────────────────┐
│  폴더 트리   │        문서 리스트 + 상세 영역          │
│              │  ┌───────────────────────────────────┐  │
│ 📂 전체문서  │  │ [업로드] [검색] [필터] [정렬]      │  │
│  ├ 인사부   │  ├───────────────────────────────────┤  │
│  ├ 재무부   │  │ 파일명 | 유형 | 크기 | 상태 | ...  │  │
│  ├ 기획부   │  │ ──────────────────────────────── │  │
│  └ 기술부   │  │ doc1.pdf | PDF | 2MB | ✅ | ...  │  │
│              │  │ doc2.hwp | HWP | 1MB | 🔄 | ...  │  │
│ 📁 개인지식  │  └───────────────────────────────────┘  │
│  └ 내문서    │  [상세 패널: 클릭한 문서의 미리보기]    │
└──────────────┴─────────────────────────────────────────┘
```

**좌측 폴더 트리 (SFR-005)**
- 조직 구조 반영: 부서별/유형별 계층 (예: 인사부 > 급여규정)
- 개인 지식영역: 사용자별 개인 문서 폴더 (권한: 본인만 접근)
- 공유 지식영역: 부서/그룹별 공유 폴더 (권한 기반 접근)
- 폴더 우클릭: 신규 폴더 생성, 이름 변경, 삭제, 권한 설정
- 드래그&드롭으로 문서 이동

**문서 리스트 테이블**
| 열 | 내용 | 기능 |
|----|------|------|
| 파일명 | 파일명 표시 (아이콘 포함) | 클릭시 상세 패널 오픈 |
| 유형 | PDF, HWP, DOCX, XLSX, TXT | 필터링 가능 |
| 크기 | MB/KB 단위 | 정렬 가능 |
| 업로드일 | YYYY-MM-DD HH:mm | 정렬 가능 |
| 상태 | ✅ 완료, 🔄 처리중, ❌ 오류 | 필터링 가능 |
| 청크수 | 청킹 결과 개수 | 품질 지표 |
| 작업 | 재처리/삭제/다운로드 | 버튼 메뉴 |

**상태 표시**
- ✅ 완료: 인제스트 성공, 벡터DB 저장 완료
- 🔄 처리중: 업로드 완료, 파싱/청킹/임베딩 진행중 (진행률 표시)
- ❌ 오류: 파싱 실패, OCR 실패 등 (에러 로그 보기 버튼)

**업로드 영역**
- 드래그&드롭 대형 박스: "파일을 여기에 드롭하세요"
- 클릭 업로드: 파일 탐색기 오픈
- 다중 파일 업로드: 최대 50개 동시 업로드
- 지원 형식: HWP, PDF, DOCX, XLS, XLSX, TXT, MD
- 업로드 진행률: 개별 파일별 프로그레스바
- 메타데이터 입력 옵션: 부서, 유형, 태그 (업로드 후 자동 추출도 가능)

**문서 상세 패널 (우측 슬라이드 패널)**
- 문서 정보:
  - 파일명, 크기, 업로드자, 업로드일시
  - 메타데이터: 제목, 작성자, 작성일, 부서, 태그
  - 청크 통계: 총 청크수, 평균 길이, 중복 청크수
- 청크 미리보기:
  - 청크 리스트 (페이지네이션)
  - 각 청크: 텍스트 내용, 임베딩 벡터 차원, 메타데이터
  - 청크 수정: 텍스트 편집 → 재임베딩 버튼
- 액션 버튼:
  - 재처리: 파싱부터 다시 실행 (설정 변경 후)
  - 메타데이터 편집: 수동으로 태그/부서 수정
  - 삭제: 벡터DB에서 완전 제거 (확인 다이얼로그)
  - 다운로드: 원본 파일 다운로드

**개인 지식영역 탭 (SFR-005)**
- 탭 전환: [전체 문서] [개인 지식]
- 개인 지식 탭:
  - 본인이 업로드한 문서만 표시
  - 다른 사용자는 접근 불가
  - 사용 사례: 개인 메모, 참고자료, 학습 문서

**검색 및 필터**
- 검색창: 파일명, 내용 전체 텍스트 검색
- 필터:
  - 유형: PDF, HWP, DOCX, XLSX, TXT
  - 상태: 완료, 처리중, 오류
  - 부서: 인사, 재무, 기획, 기술 등
  - 업로드일: 오늘, 최근 7일, 최근 30일, 사용자 정의
- 정렬: 파일명, 크기, 업로드일, 청크수 (오름차순/내림차순)

---

### 8.3 AdminPage (관리자 포털 - SFR-014)

**탭 구조**
```
[모델 관리] [프롬프트 관리] [사용자/그룹] [지식영역] [필터 설정]
```

**8.3.1 모델 관리 탭**
- LLM 목록 테이블:
  | 프로바이더 | 모델명 | 상태 | Temperature | Max Tokens | System Message |
  |-----------|--------|------|-------------|-----------|---------------|
  | Ollama | qwen2.5:7b | ✅ 활성 | 0.7 | 2048 | "당신은..." | [편집] |
  | OpenAI | gpt-4o | ✅ 활성 | 0.5 | 4096 | "You are..." | [편집] |
  | Claude | claude-3-5-sonnet | 🔴 비활성 | 0.7 | 8192 | "You are..." | [편집] |

- 모델 추가 버튼: 프로바이더 선택 → 모델명 입력 → API 키 설정
- 모델 활성화/비활성화: 토글 스위치 (비활성화된 모델은 사용자 UI에서 숨김)
- 설정 편집:
  - Temperature (0.0~2.0 슬라이더)
  - Max Tokens (256~32768)
  - System Message (멀티라인 텍스트 에디터)
  - Top P, Frequency Penalty, Presence Penalty (고급 옵션)
- 테스트 버튼: 샘플 쿼리 입력 → 모델 응답 확인

**8.3.2 프롬프트 관리 탭 (SFR-003)**
- 프롬프트 템플릿 목록:
  - RAG 시스템 프롬프트
  - 답변 생성 프롬프트
  - 재작성 프롬프트 (쿼리 개선)
  - 에이전트 라우터 프롬프트
  - 부적절 프롬프트 판별 프롬프트
- 프롬프트 편집기:
  - 코드 에디터 (Syntax Highlighting)
  - 변수 삽입: `{context}`, `{query}`, `{history}` 등
  - 미리보기: 샘플 데이터로 렌더링 결과 확인
- 버전 이력:
  - 수정 일시, 수정자, 변경 내용 (diff 보기)
  - 롤백 버튼: 이전 버전으로 복원
- 부적절 프롬프트 필터 설정:
  - 금지 키워드 리스트: 정치, 종교, 비속어 등
  - 패턴 매칭 규칙: 정규표현식 입력
  - 필터 액션: 차단 (에러 반환) 또는 경고 (로그만 기록)

**8.3.3 사용자/그룹 관리 탭**
- 사용자 목록 테이블:
  | 사용자명 | 이메일 | 역할 | 그룹 | 상태 | 등록일 | [작업] |
  |---------|-------|------|------|------|--------|--------|
  | 홍길동 | hong@example.com | 일반 | 인사부 | ✅ 활성 | 2024-01-15 | [편집][삭제] |
  | 김철수 | kim@example.com | 관리자 | 전체 | ✅ 활성 | 2024-01-10 | [편집][삭제] |

- 역할 유형:
  - 관리자: 모든 기능 접근, 설정 변경 가능
  - 일반 사용자: 채팅, 문서 업로드, 본인 문서 관리
  - 뷰어: 채팅만 가능 (문서 업로드 불가)
- 그룹 관리:
  - 그룹 CRUD: 생성, 편집, 삭제
  - 그룹별 권한: 접근 가능한 지식영역 설정
  - 사용자 그룹 할당: 다중 그룹 가능

**8.3.4 지식영역 관리 탭 (SFR-005)**
- 지식영역 목록:
  | 영역명 | 설명 | 문서 범위 | 접근 권한 | [작업] |
  |-------|------|----------|----------|--------|
  | 인사 규정 | 인사 관련 문서 | /인사부/* | 인사부 그룹 | [편집][삭제] |
  | 전사 공통 | 모든 직원 열람 가능 | /공통/* | 전체 | [편집][삭제] |

- 지식영역 생성/편집:
  - 영역명, 설명
  - 문서 범위: 폴더 경로 또는 태그 기반 선택
  - 접근 권한: 그룹 또는 사용자 개별 지정
  - 우선순위: 중복 문서가 여러 영역에 속할 때 순서

**8.3.5 필터 설정 탭**
- 부적절 프롬프트 필터:
  - 활성화/비활성화 토글
  - 금지 키워드 목록 편집 (CSV 업로드 또는 직접 입력)
  - 차단 로그 조회: 차단된 쿼리 이력, 사용자, 시각
- 콘텐츠 안전 필터:
  - 외부 API 연동 (예: OpenAI Moderation API)
  - 필터링 수준: 낮음, 중간, 높음

---

### 8.4 MonitorPage (모니터링 대시보드)

**레이아웃: 그리드 기반 대시보드**
```
┌─────────────────┬─────────────────┬─────────────────┐
│  실시간 사용통계 │   응답 시간     │   모델별 사용량  │
├─────────────────┴─────────────────┴─────────────────┤
│            RAG 품질 대시보드 (SFR-008)               │
├──────────────────────────────────────────────────────┤
│                 로그 뷰어 (SFR-010)                  │
├─────────────────┬─────────────────┬─────────────────┤
│  피드백 현황     │   Golden Data   │   API 사용량    │
└─────────────────┴─────────────────┴─────────────────┘
```

**8.4.1 실시간 사용 통계**
- 메트릭 카드:
  - 오늘 총 질문 수: 1,234건
  - 활성 사용자 수: 56명
  - 평균 응답 시간: 2.3초
  - 오류율: 0.5%
- 시간별 질문 추이 그래프 (최근 24시간)
- 실시간 업데이트: 30초마다 자동 갱신

**8.4.2 응답 시간 분석**
- 응답 시간 히스토그램: <1s, 1-3s, 3-5s, >5s
- P50, P90, P99 지연시간 표시
- 느린 쿼리 목록: 응답 시간 >5초인 쿼리 리스트 (쿼리 텍스트, 시각, 원인)

**8.4.3 모델별 사용량**
- 파이 차트: 각 LLM 모델의 사용 비율
  - qwen2.5:7b: 45%
  - gpt-4o-mini: 30%
  - claude-3-5-sonnet: 25%
- 모델별 평균 응답 시간, 오류율 비교 테이블

**8.4.4 RAG 품질 대시보드 (SFR-008)**
- 청크 품질 메트릭:
  - 평균 청크 길이: 512 토큰
  - 청크 길이 분포: 히스토그램 (100-1000 토큰 범위)
  - 중복 청크 감지: 전체 청크의 2.5% 중복 (임계값: 5%)
- 임베딩 성공/실패율:
  - 성공: 98.5%
  - 실패: 1.5% (에러 로그 보기 버튼)
- 검색 정확도:
  - 평균 유사도 점수: 0.82
  - 신뢰도 분포: 높음 60%, 중간 30%, 낮음 10%
- 재처리 알림: 청크 품질 저하 감지시 자동 알림

**8.4.5 로그 뷰어 (SFR-010)**
- 탭 구조: [질의 이력] [접속 로그] [작업 로그] [에러 로그]
- 질의 이력:
  | 시각 | 사용자 | 쿼리 | 모델 | 응답시간 | 신뢰도 | 피드백 |
  |------|--------|------|------|---------|--------|--------|
  | 14:32 | 홍길동 | "예산 기준?" | qwen2.5 | 2.1s | 0.89 | 👍 |

- 접속 로그:
  | 시각 | 사용자 | IP 주소 | 액션 | 세부사항 |
  |------|--------|---------|------|---------|
  | 14:30 | 홍길동 | 192.168.1.100 | 로그인 | 성공 |

- 작업 로그:
  | 시각 | 작업 유형 | 상태 | 대상 | 수행자 |
  |------|----------|------|------|--------|
  | 14:25 | 문서 업로드 | ✅ | doc.pdf | 김철수 |
  | 14:20 | 재처리 | 🔄 | report.hwp | 시스템 |

- 에러 로그:
  | 시각 | 에러 유형 | 메시지 | 스택 트레이스 |
  |------|----------|--------|--------------|
  | 14:15 | OCR 실패 | "Upstage API timeout" | [보기] |

- 필터 및 검색:
  - 날짜 범위: 오늘, 최근 7일, 최근 30일, 사용자 정의
  - 사용자 필터: 특정 사용자만 표시
  - 검색: 쿼리 텍스트, 에러 메시지 전체 검색
- 내보내기: CSV/JSON 형식으로 로그 다운로드

**8.4.6 피드백 현황**
- 피드백 집계:
  - 총 피드백 수: 1,024건
  - 좋아요: 892건 (87%)
  - 싫어요: 132건 (13%)
- 시간별 피드백 추이 그래프
- 부정 피드백 목록:
  - 쿼리, 답변, 사용자 보정 내용, 상태 (검토중/처리완료)
  - 검토 버튼: Golden Data 승인 또는 프롬프트 개선으로 연결

**8.4.7 Golden Data 현황**
- Golden Data 통계:
  - 총 Golden Data 수: 156쌍
  - 이번 주 추가: 12쌍
- Golden Data 목록:
  | 질문 | 답변 | 출처 | 추가일 | [작업] |
  |------|------|------|--------|--------|
  | "예산 편성 기준?" | "전년도 대비 5%..." | budget.pdf | 2024-01-20 | [편집][삭제] |

- Golden Data 활용:
  - Few-shot 예시로 프롬프트에 자동 삽입
  - 벡터DB에 높은 가중치로 저장

**8.4.8 API 사용량**
- 엔드포인트별 호출 통계:
  | 엔드포인트 | 호출 수 | 평균 응답 시간 | 에러율 |
  |-----------|--------|--------------|--------|
  | POST /chat | 1,234 | 2.3s | 0.5% |
  | POST /documents/upload | 56 | 5.1s | 2.0% |

- 피크 시간대 분석: 시간대별 요청 히트맵 (9-11시, 14-16시 피크)
- 외부 API 사용량:
  - Upstage OCR: 45건 (비용: ₩12,000)
  - OpenAI API: 234건 (토큰: 125,000)

---

### 8.5 AgentBuilderPage (에이전트 빌더 - SFR-018)

**레이아웃 구조**
```
┌────────────┬────────────────────────────────┬──────────┐
│ 노드 팔레트 │      비주얼 캔버스              │ 속성패널  │
│            │  ┌──────┐                      │          │
│ 🔹 LLM     │  │ Start│                      │ [선택 노드│
│ 🔹 RAG     │  └──┬───┘                      │  속성]   │
│ 🔹 Tool    │     │                          │          │
│ 🔹 Condition│    ▼                          │  모델:   │
│ 🔹 Transform│  ┌────┐   ┌────┐             │  [선택]  │
│            │  │RAG │──>│LLM │              │          │
│ [+ 도구등록]│  └────┘   └────┘             │  온도:   │
│            │                                │  [슬라이더]│
└────────────┴────────────────────────────────┴──────────┘
```

**좌측 노드 팔레트**
- 드래그 가능한 노드 유형:
  - 🔹 **LLM 노드**: LLM 호출 (모델, 프롬프트, 온도 설정)
  - 🔹 **RAG 노드**: 벡터DB 검색 (쿼리, top_k, 필터)
  - 🔹 **Tool 노드**: MCP 도구 실행 (도구 선택, 파라미터)
  - 🔹 **Condition 노드**: 조건 분기 (if-else 로직)
  - 🔹 **Transform 노드**: 데이터 변환 (JSON 파싱, 텍스트 추출)
  - 🔹 **Start/End 노드**: 워크플로우 시작/종료
- 각 노드: 아이콘, 이름, 간단한 설명 표시

**중앙 비주얼 캔버스 (React Flow 기반)**
- 드래그&드롭으로 노드 배치
- 노드 간 연결: 출력 포트 → 입력 포트 드래그
- 노드 스타일:
  - LLM: 파란색
  - RAG: 녹색
  - Tool: 주황색
  - Condition: 보라색
  - Transform: 회색
- 실행 시각화:
  - 실행 중인 노드: 깜빡이는 테두리
  - 완료된 노드: 체크마크 표시
  - 에러 노드: 빨간색 테두리
- 줌/팬: 마우스 휠로 줌, 드래그로 이동
- 미니맵: 우측 하단에 전체 워크플로우 미니맵

**우측 속성 패널**
- 노드 선택시 해당 노드의 속성 편집:
  - **LLM 노드**:
    - 모델 선택: qwen2.5, gpt-4o, claude-3-5-sonnet
    - 프롬프트 템플릿: "사용자 입력: {input}\n답변:"
    - 온도: 0.0~2.0 슬라이더
    - Max Tokens: 256~8192
  - **RAG 노드**:
    - 검색 쿼리: 이전 노드 출력 또는 고정 텍스트
    - Top K: 1~10
    - 필터: 부서, 날짜 범위 등
    - 유사도 임계값: 0.0~1.0
  - **Tool 노드**:
    - 도구 선택: 드롭다운 (등록된 MCP 도구 목록)
    - 파라미터: JSON 에디터 또는 폼
    - 타임아웃: 초 단위
  - **Condition 노드**:
    - 조건식: JavaScript 표현식 (예: `output.score > 0.8`)
    - True 경로, False 경로
  - **Transform 노드**:
    - 변환 타입: JSON 파싱, 텍스트 추출, 정규표현식
    - 변환 스크립트: JavaScript 코드

**도구 등록 패널 (하단 슬라이드업)**
- 코드 기반 등록:
  ```python
  # Python 함수를 MCP 도구로 등록
  def get_weather(city: str) -> str:
      # API 호출 로직
      return f"{city}의 날씨는 맑음"
  ```
  - 함수 코드 입력 → 파라미터 스키마 자동 추출 → 등록
- 노코드 등록:
  - API Endpoint URL 입력
  - HTTP Method: GET, POST, PUT, DELETE
  - 헤더: Authorization, Content-Type 등
  - 파라미터: Query, Body, Path 파라미터 정의
  - 응답 파싱: JSON Path 또는 정규표현식
- 도구 테스트: 샘플 입력으로 도구 실행 → 결과 확인

**에이전트 목록 (좌측 사이드바, 접기 가능)**
- 저장된 워크플로우 목록:
  - 이름, 설명, 생성일, 실행 횟수
  - 클릭시 캔버스에 로드
  - 복제, 삭제, 공유 버튼
- 새 워크플로우 생성: 빈 캔버스 시작

**실행 및 디버깅**
- 실행 버튼: 워크플로우 전체 실행
- 입력 데이터: JSON 형식으로 시작 노드 입력 제공
- 실행 결과:
  - 각 노드의 입력/출력 표시
  - 실행 시간, 상태 (성공/실패)
  - 에러 메시지 및 스택 트레이스
- 단계별 실행: 노드 하나씩 실행하며 디버그
- 실행 이력:
  - 과거 실행 기록 조회
  - 입력/출력 데이터 보관
  - 재실행 버튼

**MCP 서버 상태 모니터링**
- MCP 서버 리스트:
  | 서버명 | URL | 상태 | 도구 수 | [작업] |
  |-------|-----|------|--------|--------|
  | 로컬 MCP | localhost:3000 | 🟢 온라인 | 12 | [재시작][로그] |
  | 외부 MCP | api.example.com | 🔴 오프라인 | 8 | [재시작][로그] |

- 서버별 도구 목록: 클릭시 해당 서버의 도구 리스트 표시
- 연결 테스트: 서버 헬스체크 실행

**저장 및 배포**
- 저장: 워크플로우를 JSON 형식으로 저장 (`.omc/agents/{name}.json`)
- 내보내기: 워크플로우를 파일로 다운로드 (공유 가능)
- 가져오기: JSON 파일 업로드하여 워크플로우 로드
- 배포: 워크플로우를 API 엔드포인트로 노출 (향후 확장)

---

## 9. 에이전트 아키텍처 (상세 - 최우선)

### 9.1 아키텍처 개요

**전체 시스템 다이어그램**
```
┌─────────────────────────────────────────────────────────────┐
│                     프론트엔드 (React)                        │
│  ChatPage | DocumentsPage | AdminPage | MonitorPage | Builder│
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────▼───────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Agent Gateway API                         │ │
│  │  /chat, /documents, /admin, /agent/execute             │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │                                   │
│  ┌───────────────────────▼────────────────────────────────┐ │
│  │              Agent Orchestration Layer                  │ │
│  │  ┌─────────┐   ┌──────────┐   ┌───────────────────┐  │ │
│  │  │ Router  │──>│ Planner  │──>│    Executor       │  │ │
│  │  │ (분류)   │   │ (계획)    │   │    (실행)         │  │ │
│  │  └─────────┘   └──────────┘   └───────────────────┘  │ │
│  │       │              │                 │              │ │
│  │       └──────────────┼─────────────────┘              │ │
│  │                      ▼                                │ │
│  │  ┌────────────────────────────────────────────────┐  │ │
│  │  │          Memory Manager (SQLite)               │  │ │
│  │  │  - 대화 이력 저장 (세션별)                      │  │ │
│  │  │  - 컨텍스트 윈도우 관리 (MAX_HISTORY=5)        │  │ │
│  │  │  - 쿼리 재작성 (대명사 해소)                    │  │ │
│  │  └────────────────────────────────────────────────┘  │ │
│  └──────────────────────┬────────────────────────────────┘ │
│                         │                                   │
│  ┌──────────────────────┼──────────────────────────────┐   │
│  │  ┌──────────────┐   ┌▼────────────┐   ┌──────────┐ │   │
│  │  │  LLM Hub     │   │ MCP Server  │   │ Tool     │ │   │
│  │  │  (멀티 LLM)   │   │ (프로토콜)   │   │ Registry │ │   │
│  │  │              │   │             │   │          │ │   │
│  │  │ • Ollama     │   │ • DB 조회   │   │ • 동적   │ │   │
│  │  │ • OpenAI     │   │ • API 호출  │   │   등록   │ │   │
│  │  │ • Anthropic  │   │ • 파일처리  │   │ • 코드   │ │   │
│  │  │              │   │ • 계산      │   │ • 노코드 │ │   │
│  │  └──────────────┘   └─────────────┘   └──────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Agent Builder (워크플로우 설계/실행)          │   │
│  │  • DAG 기반 실행 엔진                                 │   │
│  │  • 노드: LLM, RAG, Tool, Condition, Transform        │   │
│  │  • 실행 이력 추적 및 디버깅                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                   RAG Pipeline                               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Document   │─>│  Chunker    │─>│  Embedding           │ │
│  │ Loader     │  │  (Semantic) │  │  (Upstage Solar)     │ │
│  └────────────┘  └─────────────┘  └──────────────────────┘ │
│                                              │               │
│  ┌────────────────────────────────────────────▼────────────┐│
│  │            ChromaDB (Vector Store)                      ││
│  │  • 벡터 검색 + BM25 하이브리드                          ││
│  │  • Re-ranking (Cross-encoder)                          ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**계층별 책임**
| 계층 | 책임 | 기술 스택 |
|------|------|----------|
| **프론트엔드** | 사용자 인터페이스, 실시간 스트리밍 | React, TypeScript, shadcn/ui, SSE |
| **Agent Gateway** | API 라우팅, 인증, 요청 검증 | FastAPI, Pydantic |
| **Agent Orchestration** | 질문 분류, 작업 계획, 실행 관리 | Python, LangGraph (선택) |
| **LLM Hub** | 다중 LLM 통합, 프롬프트 관리 | Ollama, OpenAI SDK, Anthropic SDK |
| **MCP Server** | 도구 실행, 컨텍스트 통합 | MCP Protocol, Python |
| **RAG Pipeline** | 문서 처리, 임베딩, 검색 | Upstage, ChromaDB, Sentence Transformers |

---

### 9.2 Router (질문 유형 분류)

**역할**
사용자 질문을 분석하여 적절한 처리 경로로 라우팅.

**분류 카테고리**
| 카테고리 | 설명 | 예시 | 실행 경로 |
|---------|------|------|----------|
| `document_search` | 문서 기반 질문 | "예산 편성 기준은?" | RAG 검색 → LLM |
| `general_query` | 일반 지식 질문 | "파이썬이 뭐야?" | LLM 직접 호출 |
| `complex_task` | 다단계 복합 질문 | "예산 비교 후 보고서 작성해줘" | Planner → 다단계 실행 |
| `tool_required` | 도구 실행 필요 | "DB에서 사용자 목록 조회" | MCP Tool → LLM |
| `chitchat` | 잡담 | "안녕하세요" | 간단한 응답 |

**구현 방식**
```python
class Router:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        사용자 질문을 다음 카테고리로 분류하세요:
        1. document_search: 문서 검색이 필요한 질문
        2. general_query: 일반 상식 또는 지식 질문
        3. complex_task: 여러 단계가 필요한 복잡한 작업
        4. tool_required: 외부 도구 실행 필요
        5. chitchat: 인사, 잡담

        JSON 형식으로 응답: {"category": "...", "confidence": 0.95}
        """

    def classify(self, query: str) -> Dict[str, Any]:
        # LLM 호출로 분류
        response = self.llm.generate(
            system=self.system_prompt,
            user=f"질문: {query}"
        )
        return json.loads(response)

    def route(self, query: str, category: str):
        if category == "document_search":
            return self._handle_rag(query)
        elif category == "complex_task":
            return self._handle_planning(query)
        elif category == "tool_required":
            return self._handle_tool(query)
        else:
            return self._handle_direct_llm(query)
```

**라우팅 시각화**
프론트엔드에 라우팅 결과를 실시간 전송 (SSE):
```json
{
  "event": "routing",
  "data": {
    "category": "document_search",
    "confidence": 0.92,
    "plan": "RAG 검색 후 답변 생성"
  }
}
```

---

### 9.3 Planner (작업 계획)

**역할**
복합 질문을 sub-task로 분해하고 실행 순서를 결정.

**작업 분해 예시**
- 사용자 질문: "2023년과 2024년 예산을 비교하고 증감률을 계산한 후 보고서를 작성해줘"
- Sub-tasks:
  1. 2023년 예산 문서 검색
  2. 2024년 예산 문서 검색
  3. 두 문서에서 예산 수치 추출
  4. 증감률 계산 (Tool: Calculator)
  5. 보고서 생성 (LLM)

**의존성 그래프**
```
Task 1 ──┐
         ├──> Task 3 ──> Task 4 ──> Task 5
Task 2 ──┘
```

**구현**
```python
class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str) -> List[Task]:
        prompt = f"""
        다음 질문을 단계별 작업(sub-task)으로 분해하세요:
        질문: {query}

        각 작업은 다음 형식으로 작성:
        - task_id: 고유 ID
        - description: 작업 설명
        - type: rag_search, tool_call, llm_generate
        - dependencies: 의존하는 task_id 리스트
        - params: 실행에 필요한 파라미터

        JSON 배열로 응답하세요.
        """
        response = self.llm.generate(prompt)
        tasks = json.loads(response)
        return [Task(**t) for t in tasks]

    def execute_plan(self, tasks: List[Task]):
        results = {}
        for task in self._topological_sort(tasks):
            # 의존성 결과를 파라미터에 주입
            params = self._inject_dependencies(task, results)
            result = self._execute_task(task, params)
            results[task.task_id] = result
        return results
```

---

### 9.4 Memory (대화 메모리)

**역할**
세션별 대화 이력 관리 및 컨텍스트 윈도우 유지.

**SQLite 스키마**
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' 또는 'assistant'
    content TEXT NOT NULL,
    metadata JSON,  -- 출처, 신뢰도 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session ON conversations(session_id, created_at);
```

**컨텍스트 윈도우 관리**
```python
class MemoryManager:
    MAX_HISTORY = 5  # 최근 5턴만 유지

    def get_context(self, session_id: str) -> List[Message]:
        # 최근 5턴 (user + assistant 쌍) 조회
        messages = db.query(
            "SELECT role, content FROM conversations "
            "WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, self.MAX_HISTORY * 2)
        )
        return list(reversed(messages))

    def add_message(self, session_id: str, role: str, content: str, metadata: dict):
        db.execute(
            "INSERT INTO conversations (session_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(metadata))
        )
```

**쿼리 재작성 (대명사 해소)**
```python
def rewrite_query(self, session_id: str, query: str) -> str:
    """
    이전 대화를 참고하여 대명사를 구체적인 명사로 변경.
    예: "그거 언제야?" → "2024년 예산 편성 기준은 언제야?"
    """
    context = self.get_context(session_id)
    prompt = f"""
    이전 대화:
    {self._format_context(context)}

    현재 질문: {query}

    현재 질문의 대명사(그거, 이거, 그 문서 등)를 이전 대화 내용을 바탕으로 구체적으로 바꿔주세요.
    """
    rewritten = self.llm.generate(prompt)
    return rewritten
```

---

### 9.5 LLM Hub

**역할**
다중 LLM 프로바이더를 통합하여 일관된 인터페이스 제공.

**팩토리 패턴**
```python
class LLMFactory:
    @staticmethod
    def create(provider: str, model: str, **kwargs) -> BaseLLM:
        if provider == "vllm":
            return VLLMProvider(model, **kwargs)
        elif provider == "ollama":
            return OllamaLLM(model, **kwargs)
        elif provider == "openai":
            return OpenAILLM(model, **kwargs)
        elif provider == "anthropic":
            return AnthropicLLM(model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        pass

class OllamaLLM(BaseLLM):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(base_url)

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return response['message']['content']

    def stream(self, prompt: str, system: str = None, **kwargs):
        for chunk in self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            **kwargs
        ):
            yield chunk['message']['content']

class VLLMProvider(BaseLLM):
    """vLLM OpenAI-compatible API wrapper"""
    def __init__(self, model: str, base_url: str = "http://vllm-server:8000/v1"):
        self.model = model
        self.client = httpx.Client(base_url=base_url)

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        # vLLM uses OpenAI-compatible /v1/chat/completions endpoint
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            }
        )
        return response.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str, system: str = None, **kwargs):
        # vLLM streaming via SSE
        with self.client.stream(
            "POST", "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                **kwargs
            }
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    yield chunk["choices"][0]["delta"].get("content", "")
```

**모델 전환**
프론트엔드에서 모델 선택 → API 요청에 `provider`, `model` 파라미터 포함:
```json
{
  "query": "예산 기준은?",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.7
}
```

**로드밸런싱/페일오버 (향후 확장)**
- 여러 Ollama 인스턴스에 라운드 로빈
- OpenAI API 실패시 Anthropic으로 자동 전환

---

### 9.6 MCP Server (Model Context Protocol)

**역할**
외부 도구와 LLM을 연결하여 에이전트가 실제 작업을 수행할 수 있게 함.

**MCP 프로토콜 개요**
- 표준화된 도구 호출 인터페이스
- 도구 정의: 이름, 설명, 파라미터 스키마
- 실행 결과를 LLM 컨텍스트에 자동 통합

**내장 도구 예시**
```python
# tools/database.py
@mcp_tool(
    name="query_database",
    description="SQL 쿼리를 실행하여 데이터베이스 조회",
    parameters={
        "query": {"type": "string", "description": "실행할 SQL 쿼리"},
        "limit": {"type": "integer", "default": 100}
    }
)
def query_database(query: str, limit: int = 100):
    # 안전성: SELECT만 허용
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    results = db.execute(query, limit=limit)
    return {"rows": results, "count": len(results)}

# tools/calculator.py
@mcp_tool(
    name="calculate",
    description="수식을 계산",
    parameters={
        "expression": {"type": "string", "description": "계산할 수식 (예: 2+2*3)"}
    }
)
def calculate(expression: str):
    # 안전성: eval 대신 ast.literal_eval 또는 제한된 파서 사용
    try:
        result = safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

**도구 실행 플로우**
```
1. LLM이 도구 호출 필요성 인지 (함수 호출 기능)
2. 도구 이름 + 파라미터 반환
3. MCP Server가 해당 도구 실행
4. 결과를 LLM 컨텍스트에 추가
5. LLM이 결과 기반으로 최종 답변 생성
```

**외부 MCP 서버 연결**
```python
class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def list_tools(self) -> List[ToolDefinition]:
        response = requests.get(f"{self.server_url}/tools")
        return [ToolDefinition(**t) for t in response.json()]

    def execute_tool(self, tool_name: str, params: dict):
        response = requests.post(
            f"{self.server_url}/execute",
            json={"tool": tool_name, "params": params}
        )
        return response.json()
```

---

### 9.7 Tool Registry

**역할**
도구를 동적으로 등록/관리하고, Agent Builder에서 사용 가능하게 함.

**레지스트리 구조**
```python
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register_function(self, func: Callable, metadata: dict):
        """
        Python 함수를 도구로 등록.
        함수 시그니처에서 파라미터 스키마 자동 추출.
        """
        tool_def = ToolDefinition(
            name=metadata.get("name", func.__name__),
            description=metadata.get("description", func.__doc__),
            parameters=self._extract_params(func),
            handler=func
        )
        self.tools[tool_def.name] = tool_def

    def register_api(self, config: dict):
        """
        노코드 방식: API endpoint를 도구로 등록.
        """
        tool_def = ToolDefinition(
            name=config["name"],
            description=config["description"],
            parameters=config["parameters"],
            handler=lambda **kwargs: self._call_api(config, kwargs)
        )
        self.tools[tool_def.name] = tool_def

    def _call_api(self, config: dict, params: dict):
        response = requests.request(
            method=config["method"],
            url=config["url"],
            headers=config.get("headers", {}),
            json=params
        )
        return response.json()

    def execute(self, tool_name: str, params: dict):
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        return self.tools[tool_name].handler(**params)
```

**도구 메타데이터 예시**
```json
{
  "name": "get_weather",
  "description": "도시의 현재 날씨 조회",
  "parameters": {
    "city": {
      "type": "string",
      "description": "도시 이름",
      "required": true
    }
  },
  "auth": {
    "type": "api_key",
    "header": "X-API-Key",
    "value": "${WEATHER_API_KEY}"
  }
}
```

---

### 9.8 Agent Builder

**역할**
비주얼 워크플로우 설계 및 실행.

**DAG 기반 실행 엔진**
```python
class WorkflowEngine:
    def __init__(self, registry: ToolRegistry, llm_hub: LLMFactory):
        self.registry = registry
        self.llm_hub = llm_hub

    def execute(self, workflow: Workflow, inputs: dict):
        """
        워크플로우를 DAG 위상 정렬 순서로 실행.
        """
        results = {"input": inputs}
        sorted_nodes = self._topological_sort(workflow.nodes)

        for node in sorted_nodes:
            # 의존 노드 결과를 입력으로 주입
            node_inputs = self._collect_inputs(node, results)

            # 노드 유형별 실행
            if node.type == "LLMNode":
                result = self._execute_llm(node, node_inputs)
            elif node.type == "RAGNode":
                result = self._execute_rag(node, node_inputs)
            elif node.type == "ToolNode":
                result = self._execute_tool(node, node_inputs)
            elif node.type == "ConditionNode":
                result = self._execute_condition(node, node_inputs)

            results[node.id] = result

        return results[workflow.output_node]

    def _execute_llm(self, node: LLMNode, inputs: dict):
        llm = self.llm_hub.create(node.provider, node.model)
        prompt = node.prompt_template.format(**inputs)
        return llm.generate(prompt, system=node.system_message, temperature=node.temperature)

    def _execute_tool(self, node: ToolNode, inputs: dict):
        return self.registry.execute(node.tool_name, inputs)
```

**노드 유형 정의**
```python
@dataclass
class Node:
    id: str
    type: str
    inputs: List[str]  # 연결된 입력 노드 ID

@dataclass
class LLMNode(Node):
    provider: str
    model: str
    prompt_template: str
    system_message: str
    temperature: float

@dataclass
class RAGNode(Node):
    query_template: str
    top_k: int
    filters: dict

@dataclass
class ToolNode(Node):
    tool_name: str
    params: dict

@dataclass
class ConditionNode(Node):
    condition: str  # JavaScript 표현식
    true_branch: str  # 노드 ID
    false_branch: str
```

**워크플로우 저장 형식 (JSON)**
```json
{
  "name": "예산 비교 워크플로우",
  "nodes": [
    {
      "id": "node1",
      "type": "RAGNode",
      "query_template": "2023년 예산 문서",
      "top_k": 3,
      "inputs": []
    },
    {
      "id": "node2",
      "type": "LLMNode",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "prompt_template": "다음 문서에서 예산 수치를 추출: {node1}",
      "inputs": ["node1"]
    },
    {
      "id": "node3",
      "type": "ToolNode",
      "tool_name": "calculate",
      "params": {"expression": "{node2} - {previous_budget}"},
      "inputs": ["node2"]
    }
  ],
  "output_node": "node3"
}
```

---

## 10. 품질 관리 체계

### 10.1 신뢰도 점수 시스템

**점수 계산 방법**
```python
def calculate_confidence(chunks: List[Chunk]) -> float:
    """
    검색된 청크의 유사도 점수 기반으로 신뢰도 계산.
    """
    if not chunks:
        return 0.0

    # 상위 3개 청크의 평균 유사도
    top_scores = [c.score for c in chunks[:3]]
    avg_score = sum(top_scores) / len(top_scores)

    # 점수 분포 표준편차 (일관성 지표)
    std_dev = np.std(top_scores)
    consistency_penalty = 1 - min(std_dev, 0.2)

    return avg_score * consistency_penalty
```

**신뢰도 등급**
| 등급 | 점수 범위 | 배지 | 의미 |
|------|----------|------|------|
| 높음 | 0.8 ~ 1.0 | 🟢 녹색 | 검색 문서와 높은 유사도, 답변 신뢰 가능 |
| 중간 | 0.5 ~ 0.8 | 🟡 황색 | 부분적 일치, 답변 참고용 |
| 낮음 | 0.0 ~ 0.5 | 🔴 적색 | 유사 문서 부족, 답변 불확실 |

---

### 10.2 안전장치 (SFR-012)

**낮은 신뢰도 처리**
```python
if confidence < 0.5:
    response = f"""
    ⚠️ 확실하지 않은 답변입니다.

    검색된 문서에서 명확한 정보를 찾지 못했습니다.
    다음 방법을 시도해보세요:
    1. 질문을 더 구체적으로 작성
    2. 관련 문서가 업로드되었는지 확인
    3. 관리자에게 문의

    제한적인 정보로 추정한 답변:
    {generated_answer}
    """
```

**표준 안내 문구**
- "검색된 문서가 없습니다. 다른 키워드로 시도해보세요."
- "이 질문은 업로드된 문서 범위를 벗어납니다."
- "개인정보가 포함된 질문은 답변할 수 없습니다."

---

### 10.3 Golden Data 관리

**Golden Data 생성 프로세스**
```
1. 사용자가 답변에 👍 좋아요 → Golden Data 후보로 마킹
2. 운영자가 MonitorPage에서 검토
3. 승인시:
   a. 벡터DB에 높은 가중치로 즉시 추가
   b. 프롬프트 Few-shot 예시로 등록
4. 거부시: 피드백 이유 기록 → 프롬프트 개선 참고
```

**Few-shot 예시 활용**
```python
def build_prompt_with_golden_data(query: str, context: str):
    golden_examples = db.query(
        "SELECT question, answer FROM golden_data "
        "ORDER BY similarity(embedding, ?) DESC LIMIT 2",
        embed(query)
    )

    few_shot = "\n\n".join([
        f"Q: {ex.question}\nA: {ex.answer}"
        for ex in golden_examples
    ])

    return f"""
    다음은 우수 답변 예시입니다:
    {few_shot}

    이제 다음 질문에 답하세요:
    Q: {query}
    컨텍스트: {context}
    A:
    """
```

---

### 10.4 피드백 루프

**피드백 수집 → 개선 사이클**
```
1. 사용자 피드백 (👍👎) → DB 저장
2. 주간 피드백 리포트 생성:
   - 부정 피드백 Top 10 쿼리
   - 낮은 신뢰도 답변 목록
3. 운영자 검토:
   - 프롬프트 수정 필요 여부
   - 문서 추가 필요 여부
   - Golden Data 승인
4. 개선 적용:
   - 프롬프트 버전 업데이트
   - 문서 재처리
   - Golden Data 추가
5. 벤치마크 재실행 → 정확도 변화 확인
```

---

### 10.5 벤치마크 (정량 평가)

**정답 데이터셋 구성**
- 100-200쌍 Q&A
- 실제 업무 질문 기반
- 전문가가 검증한 정답
- 다양한 난이도 분포 (쉬움 40%, 보통 40%, 어려움 20%)

**평가 지표**
```python
def evaluate_rag_system(test_set):
    metrics = {
        "recall": 0.0,        # 정답 문서를 검색했는가?
        "precision": 0.0,     # 검색 문서가 관련성 있는가?
        "semantic_similarity": 0.0,  # 답변과 정답의 의미 유사도
        "latency": 0.0,       # 평균 응답 시간
        "confidence_accuracy": 0.0  # 신뢰도와 실제 정확도 일치율
    }

    for qa in test_set:
        # 시스템 답변 생성
        answer, chunks, confidence = rag_system.query(qa.question)

        # Recall: 정답 문서가 검색됐는지
        if qa.source_document in [c.metadata['source'] for c in chunks]:
            metrics["recall"] += 1

        # Semantic Similarity: 임베딩 기반 코사인 유사도
        sim = cosine_similarity(embed(answer), embed(qa.ground_truth))
        metrics["semantic_similarity"] += sim

        # 신뢰도 정확도: 높은 신뢰도 = 높은 유사도?
        if (confidence > 0.8 and sim > 0.85) or (confidence < 0.5 and sim < 0.6):
            metrics["confidence_accuracy"] += 1

    # 평균 계산
    for key in metrics:
        metrics[key] /= len(test_set)

    return metrics
```

**목표 기준**
- Recall ≥ 90%
- Precision ≥ 90%
- Semantic Similarity ≥ 0.85
- 평균 응답 시간 ≤ 3초

---

### 10.6 모니터링 대시보드 (SFR-008)

**실시간 품질 지표**
- 청크 길이 분포: 히스토그램으로 시각화
- 중복 청크 비율: 5% 초과시 경고
- 임베딩 실패율: 1% 초과시 알림
- 평균 신뢰도 점수: 트렌드 그래프

**알림 조건**
| 지표 | 임계값 | 알림 액션 |
|------|--------|----------|
| 중복 청크 비율 | >5% | 이메일 + Slack 알림 |
| 임베딩 실패율 | >1% | 운영자 대시보드 배지 |
| 평균 응답 시간 | >5s | 성능 최적화 필요 플래그 |
| 부정 피드백 비율 | >20% | 주간 리뷰 미팅 자동 생성 |

---

## 11. 보안 고려사항

### 11.1 RBAC 기반 접근 제어 (SER-001)

**역할 정의**
| 역할 | 권한 |
|------|------|
| 관리자 | 모든 기능, 설정 변경, 사용자 관리 |
| 일반 사용자 | 채팅, 문서 업로드/조회 (권한 범위 내), 피드백 |
| 뷰어 | 채팅만 가능, 문서 업로드/수정 불가 |

**구현 (FastAPI Dependency)**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

def require_role(role: str):
    def role_checker(user = Depends(get_current_user)):
        if user.role != role and user.role != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

# API 엔드포인트에 적용
@app.post("/admin/users")
def create_user(user_data: dict, current_user = Depends(require_role("admin"))):
    # 관리자만 접근 가능
    pass
```

---

### 11.2 시큐어 코딩 (SER-002, SER-005)

**OWASP Top 10 대응**
| 위협 | 대응 방안 |
|------|----------|
| SQL Injection | Prepared Statement 사용, ORM (SQLAlchemy) |
| XSS | 입력값 HTML 이스케이프, CSP 헤더 |
| CSRF | CSRF 토큰 검증 (SameSite 쿠키) |
| Broken Auth | JWT 토큰, bcrypt 비밀번호 해싱 |
| Sensitive Data Exposure | HTTPS 강제, 환경변수로 API 키 관리 |

**입력 검증**
```python
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    query: str
    session_id: str

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 5000:
            raise ValueError('Query too long (max 5000 chars)')
        return v
```

---

### 11.3 개인정보 보호 (SER-003)

**PII 자동 감지 및 마스킹**
```python
import re

def detect_and_mask_pii(text: str) -> str:
    """
    개인정보 패턴 감지 후 마스킹.
    - 주민등록번호: 123456-1******
    - 전화번호: 010-****-5678
    - 이메일: abc***@example.com
    """
    # 주민등록번호
    text = re.sub(r'(\d{6})-(\d{7})', r'\1-*******', text)

    # 전화번호
    text = re.sub(r'(01[0-9])-(\d{3,4})-(\d{4})', r'\1-****-\3', text)

    # 이메일
    text = re.sub(r'([a-zA-Z0-9._%+-]{1,3})[a-zA-Z0-9._%+-]*(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'\1***\2', text)

    return text
```

**암호화 저장**
```python
from hashlib import sha256

def hash_sensitive_data(data: str) -> str:
    """
    민감 정보 SHA-256 해싱 (단방향).
    """
    return sha256(data.encode()).hexdigest()

# 비밀번호는 bcrypt 사용
import bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())
```

---

### 11.4 정보보호 정책 준수 (SER-004, SER-006)

**국가정보원 보안 정책**
- 소스코드 보안 취약점 점검 (연 2회)
- 접근 로그 최소 6개월 보관
- 관리자 계정 다단계 인증 (MFA)

**전자정부법 준수**
- 개인정보 처리 방침 명시
- 사용자 동의 없이 개인정보 수집 금지
- 개인정보 열람/수정/삭제 권리 보장

---

### 11.5 데모 환경 보안 설정

**환경변수 관리 (.env 파일)**
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
UPSTAGE_API_KEY=up-...
DATABASE_URL=sqlite:///./data/app.db
SECRET_KEY=your-secret-key-here

# .gitignore에 추가
.env
```

**CORS 설정**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 도메인만 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

**입력값 검증**
```python
# 파일 업로드 크기 제한
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # 파일 확장자 화이트리스트
    allowed_extensions = {'.pdf', '.hwp', '.docx', '.xlsx', '.txt'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not allowed")
```

---

## 12. 구현 로드맵 (4주)

### Week 1: 기반 구조 + 코어

**Day 1-2: 프로젝트 셋업**
- [ ] pyproject.toml 생성 (Poetry 또는 UV)
- [ ] React 프로젝트 생성 (Vite + TypeScript)
- [ ] 디렉토리 구조 설정:
  ```
  flux-rag/
  ├── backend/
  │   ├── core/           # LLM 추상화, 설정
  │   ├── pipeline/       # 문서 처리
  │   ├── rag/            # RAG 로직
  │   ├── agent/          # 에이전트 오케스트레이션
  │   ├── api/            # FastAPI 라우터
  │   └── main.py
  ├── frontend/
  │   ├── src/
  │   │   ├── components/
  │   │   ├── pages/
  │   │   └── lib/
  │   └── package.json
  └── docs/
  ```
- [ ] 환경변수 템플릿 (.env.example)
- [ ] Docker Compose 설정 (선택, ChromaDB용)

**Day 3-4: LLM 추상화 레이어**
- [ ] `core/llm/base.py`: BaseLLM 추상 클래스
- [ ] `core/llm/ollama.py`: OllamaLLM 구현
- [ ] `core/llm/openai.py`: OpenAILLM 구현
- [ ] `core/llm/anthropic.py`: AnthropicLLM 구현
- [ ] `core/llm/factory.py`: LLMFactory
- [ ] 단위 테스트 (pytest)

**Day 5: 임베딩 + ChromaDB**
- [ ] `core/embedding.py`: Upstage Solar Embedding API 연동
- [ ] `rag/vector_store.py`: ChromaDB 클라이언트, CRUD 구현
- [ ] 벡터 추가/검색 테스트

---

### Week 2: 문서 파이프라인 + RAG

**Day 1-2: 문서 로더**
- [ ] `pipeline/loaders/pdf.py`: PyPDF2 또는 pdfplumber
- [ ] `pipeline/loaders/hwp.py`: pyhwp + LibreOffice fallback
- [ ] `pipeline/loaders/docx.py`: python-docx
- [ ] `pipeline/loaders/excel.py`: openpyxl
- [ ] `pipeline/loaders/ocr.py`: Upstage OCR API 연동
- [ ] 통합 테스트 (샘플 문서 파싱)

**Day 3: 시맨틱 청킹 + 메타데이터**
- [ ] `pipeline/chunker.py`: 시맨틱 청킹 (LangChain RecursiveCharacterTextSplitter 활용)
- [ ] 청크 크기 최적화 (500-1000 토큰)
- [ ] 메타데이터 추출 (제목, 작성자, 날짜, 섹션)
- [ ] 중복 제거 로직

**Day 4: 하이브리드 검색 + Re-ranking**
- [ ] `rag/retriever.py`: 벡터 검색 (ChromaDB)
- [ ] `rag/bm25.py`: BM25 키워드 검색 (rank-bm25 라이브러리)
- [ ] `rag/hybrid.py`: 벡터 + BM25 점수 결합 (가중치 0.7:0.3)
- [ ] `rag/reranker.py`: Cross-encoder re-ranking (sentence-transformers)

**Day 5: RAG 체인 + 프롬프트 관리**
- [ ] `rag/chain.py`: 검색 → 컨텍스트 구성 → LLM 호출
- [ ] `core/prompts.py`: 프롬프트 템플릿 관리 (DB 또는 YAML)
- [ ] 신뢰도 계산 로직
- [ ] 통합 RAG 테스트 (쿼리 → 답변)

---

### Week 3: 에이전트 + API

**Day 1: Agent 코어 (Router + Planner + Memory)**
- [ ] `agent/router.py`: 질문 분류 로직
- [ ] `agent/planner.py`: Sub-task 분해
- [ ] `agent/memory.py`: SQLite 대화 이력 관리
- [ ] 단위 테스트

**Day 2: MCP 서버 + Tool Registry**
- [ ] `agent/mcp/server.py`: MCP 프로토콜 구현
- [ ] `agent/mcp/tools/`: 내장 도구 (database, calculator, file_handler)
- [ ] `agent/registry.py`: 도구 동적 등록
- [ ] 도구 실행 테스트

**Day 3: Agent Builder 백엔드**
- [ ] `agent/builder/workflow.py`: 워크플로우 정의 (Node, Edge)
- [ ] `agent/builder/engine.py`: DAG 실행 엔진
- [ ] 워크플로우 저장/로드 (JSON)
- [ ] 실행 이력 추적

**Day 4-5: FastAPI 전체 API 구현**
- [ ] `api/chat.py`: POST /chat (스트리밍 응답 SSE)
- [ ] `api/documents.py`: 문서 CRUD, 업로드, 재처리
- [ ] `api/admin.py`: 모델/프롬프트/사용자 관리
- [ ] `api/monitor.py`: 로그/통계 조회
- [ ] `api/agent.py`: 에이전트 워크플로우 실행
- [ ] API 문서 (Swagger UI) 검토

---

### Week 4: 프론트엔드 + 통합 테스트

**Day 1-2: ChatPage + DocumentsPage**
- [ ] ChatPage: 채팅 UI, SSE 스트리밍, 마크다운 렌더링
- [ ] 출처 문서 모달, 피드백 버튼
- [ ] DocumentsPage: 폴더 트리, 문서 리스트, 업로드 UI
- [ ] 문서 상세 패널

**Day 3: AdminPage + MonitorPage**
- [ ] AdminPage: 모델/프롬프트 편집 UI, 사용자 관리
- [ ] MonitorPage: 대시보드 차트 (Recharts), 로그 테이블

**Day 4: P0 기능 안정화 + 시연 시나리오 최적화**
- [ ] E2E 통합 테스트 (채팅 → RAG → 출처 확인 → 피드백)
- [ ] 성능 최적화 (응답 시간 < 3초 목표)
- [ ] 에러 핸들링 강화 (타임아웃, 재시도, 사용자 친화적 메시지)
- [ ] 시연 시나리오 리허설 (스크립트 작성)

**Day 5: 통합 테스트 + 더미 데이터 + 시연 준비**
- [ ] E2E 테스트: 문서 업로드 → 질의응답 → 출처 확인
- [ ] 더미 데이터 생성 (샘플 문서 10개, 사용자 계정)
- [ ] 성능 벤치마크 실행
- [ ] 시연 시나리오 작성 및 리허설

### 시연 데이터 전략

**Phase 1 (현재 ~ 3/5 전): 가스안전 일반 데이터**
- **데이터 소스**: 한국가스안전공사 공개 자료 크롤링
  - 가스안전 규정, 매뉴얼, 보고서
  - 가스안전관리법 관련 공개 문서
  - 일반 가스 안전 가이드라인
- **용도**:
  - RAG 파이프라인 검증 (청킹, 임베딩, 검색 정확도)
  - 시스템 통합 테스트 (문서 업로드 → 질의응답)
  - 초기 시연 시나리오 구성
- **규모**: 약 50-100개 문서, 2-3GB

**Phase 2 (3/5 이후): 실제 도메인 데이터**
- **데이터 소스**: 한국가스기술공사 실제 문서 수령 (3월 5일 예정)
  - 내부 기술 문서, 프로젝트 보고서
  - 도메인 특화 자료 (비공개)
- **전환 작업**:
  - Phase 1 데이터 기반 파이프라인 재사용
  - 도메인 특화 청킹 전략 튜닝
  - 실제 업무 시나리오로 시연 스크립트 교체
- **리스크 대응**:
  - Phase 2 데이터 형식이 예상과 다를 경우 파서 수정
  - 민감 정보 포함 시 PII 마스킹 강화
  - 3/5 이후 1주일 버퍼 확보 (데이터 정제 및 시나리오 재작성)

**참고사항**
- Phase 2 데이터에 따라 시연 시나리오 및 더미 문서 변경 가능
- 두 데이터셋 모두 동일 파이프라인 사용 → 데이터 교체 비용 최소화
- Phase 1 공개 데이터는 개발/테스트 환경에서 지속 사용

---

### P2 (향후 구현)

**Agent Builder 비주얼 캔버스**
- React Flow 기반 워크플로우 시각적 편집기
- 노드 팔레트 (RAG, 계산기, API 호출 등)
- 속성 패널 (노드 설정, 파라미터 조정)
- 실행 시각화 (실시간 상태 표시)
- 워크플로우 저장/로드/공유
- **우선순위**: 1차 시연에서 제외, 피드백 반영 후 1개월 내 추가 구현 가능
- **근거**: MCP 서버 + Tool Registry + Agent 워크플로우 실행 엔진(P0)만으로도 에이전트 기능 시연 가능. UI는 차후 UX 개선 단계에서 추가.

---

## 13. 검증 계획

### 13.1 단위 테스트 (pytest)

**테스트 범위**
| 모듈 | 테스트 항목 |
|------|-----------|
| `core/llm/` | 각 LLM 프로바이더 응답 생성, 스트리밍, 에러 핸들링 |
| `pipeline/loaders/` | PDF/HWP/DOCX/XLSX 파싱, OCR 성공/실패 |
| `pipeline/chunker.py` | 청킹 결과 길이, 메타데이터 추출, 중복 제거 |
| `rag/retriever.py` | 벡터 검색, 하이브리드 검색, Re-ranking 점수 |
| `agent/router.py` | 질문 분류 정확도 (샘플 50개) |
| `agent/memory.py` | 대화 저장/조회, 컨텍스트 윈도우 |

**실행**
```bash
pytest backend/tests/ -v --cov=backend --cov-report=html
```

---

### 13.2 API 테스트

**FastAPI TestClient**
```python
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={
        "query": "예산 편성 기준은?",
        "session_id": "test-session",
        "provider": "ollama",
        "model": "qwen2.5:7b"
    })
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["confidence"] > 0.0
```

**Swagger UI 수동 테스트**
- `http://localhost:8000/docs`에서 모든 엔드포인트 테스트
- 각 API의 요청/응답 스키마 확인

---

### 13.3 E2E 테스트

**시나리오: 전체 플로우**
1. 문서 업로드: `POST /documents/upload` (sample.pdf)
2. 인제스트 대기: 상태 확인 (`GET /documents/{id}`) → "completed"
3. 질의응답: `POST /chat` ("sample.pdf의 주요 내용은?")
4. 출처 확인: 응답의 `sources` 필드에 sample.pdf 포함 여부
5. 피드백: `POST /chat/feedback` (좋아요)

**자동화 (Playwright 또는 Selenium)**
```javascript
// E2E 테스트 (Playwright)
test('문서 업로드 및 질의응답', async ({ page }) => {
  await page.goto('http://localhost:3000/documents');
  await page.setInputFiles('input[type="file"]', 'sample.pdf');
  await page.waitForSelector('text=✅ 완료');

  await page.goto('http://localhost:3000/chat');
  await page.fill('textarea', 'sample.pdf의 내용은?');
  await page.click('button:has-text("전송")');

  await page.waitForSelector('.message-card');
  const sources = await page.textContent('.sources');
  expect(sources).toContain('sample.pdf');
});
```

---

### 13.4 성능 벤치마크

**100-200쌍 Q&A 정답셋 준비**
```json
[
  {
    "question": "2024년 예산 편성 기준은?",
    "ground_truth": "전년도 대비 5% 증액 원칙...",
    "source_document": "2024_예산편성지침.pdf"
  },
  ...
]
```

**평가 스크립트 실행**
```python
results = evaluate_rag_system(load_test_set("benchmark.json"))
print(f"Recall: {results['recall']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Semantic Similarity: {results['semantic_similarity']:.2f}")
print(f"Avg Latency: {results['latency']:.2f}s")
```

**목표 달성 여부 확인**
- Recall ≥ 90%: ✅ 또는 ❌
- Precision ≥ 90%: ✅ 또는 ❌
- Semantic Similarity ≥ 0.85: ✅ 또는 ❌

---

### 13.5 모델 교체 테스트

**동일 질문, 다른 모델 비교**
```python
models = [
    ("ollama", "qwen2.5:7b"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-20241022")
]

query = "2024년 예산 편성 기준은?"

for provider, model in models:
    answer, confidence, latency = rag_system.query(query, provider, model)
    print(f"{provider}/{model}: {confidence:.2f}, {latency:.2f}s")
    print(f"  Answer: {answer[:100]}...")
```

**비교 지표**
- 답변 품질 (수동 평가 또는 Semantic Similarity)
- 응답 시간
- 비용 (API 호출 기준)

---

### 13.6 안전장치 테스트

**불확실 질문 입력**
```python
# 벡터DB에 관련 문서가 없는 질문
query = "양자역학의 기본 원리는?"

answer, confidence = rag_system.query(query)

assert confidence < 0.5, "신뢰도가 낮아야 함"
assert "확실하지 않은 답변입니다" in answer, "표준 응답 포함 확인"
```

---

### 13.7 에이전트 테스트

**MCP 도구 실행**
```python
# Calculator 도구 테스트
result = registry.execute("calculate", {"expression": "2+2*3"})
assert result["result"] == 8

# Database 조회 도구 테스트
result = registry.execute("query_database", {"query": "SELECT * FROM users LIMIT 5"})
assert len(result["rows"]) <= 5
```

**Agent Builder 워크플로우 실행**
```python
workflow = load_workflow("workflows/budget_comparison.json")
inputs = {"year1": "2023", "year2": "2024"}

results = workflow_engine.execute(workflow, inputs)

assert "comparison_result" in results
assert results["comparison_result"]["difference"] > 0
```

---

## 14. 리스크 및 대응방안

| 리스크 | 영향도 | 확률 | 대응방안 |
|--------|--------|------|---------|
| **HWP 파싱 실패** | 높음 | 중간 | • pyhwp 1차 시도 <br> • 실패시 LibreOffice CLI로 변환 (hwp → docx) <br> • 최악의 경우 OCR 폴백 (Upstage) |
| **LLM 성능 요구사항** | 중간 | 낮음 | • vLLM을 주 LLM 서빙 엔진으로 사용 (시연: RunPod, 운영: on-premise H200) <br> • Ollama는 개발/테스트 용도로만 사용 <br> • LLM 추상화 계층으로 프로바이더 교체 용이 <br> • OpenAI/Claude로 즉시 전환 가능 |
| **ChromaDB 대규모 데이터 성능** | 중간 | 중간 | • ChromaDB는 시연/개발 단계용, 운영은 Milvus로 마이그레이션 계획 <br> • Milvus: 분산 벡터DB, GPU 가속 (L40S 활용), 엔터프라이즈급 <br> • 동일 임베딩 모델(bge-m3) 사용하여 벡터 호환성 보장 <br> • VectorStore 추상화 계층으로 교체 용이 |
| **1개월 일정 초과** | 높음 | 중간 | • P0 기능 우선 구현 (채팅, RAG, 문서 관리) <br> • Agent Builder는 최소 MVP (노드 3-4개) <br> • 프론트엔드 일부 기능 생략 (피드백 UI 등) <br> • 주간 마일스톤 체크 및 우선순위 재조정 |
| **Upstage OCR API 비용/속도** | 낮음 | 낮음 | • 이미지 기반 문서만 OCR 적용 <br> • 캐싱: 동일 문서 재업로드시 OCR 재사용 <br> • 배치 처리: 비동기 큐로 API 호출 분산 <br> • 월간 예산 한도 설정 및 모니터링 |
| **망분리 환경 라이브러리 설치** | 중간 | 중간 | • 사전 준비: pip download로 .whl 패키지 번들 생성 <br> • Docker 이미지 빌드 후 tar 파일로 전달 <br> • 오프라인 설치 가이드 문서화 |
| **React 프론트엔드 복잡도** | 중간 | 낮음 | • shadcn/ui 컴포넌트 라이브러리로 개발 속도 향상 <br> • React Flow (Agent Builder)는 검증된 라이브러리 <br> • 복잡한 상태관리는 Zustand로 단순화 |
| **LLM 환각(Hallucination)** | 높음 | 중간 | • 신뢰도 점수로 불확실성 표시 <br> • 출처 문서 명시로 검증 가능성 확보 <br> • Golden Data로 Few-shot 학습 강화 <br> • 사용자 피드백으로 지속 개선 |
| **개인정보 유출** | 높음 | 낮음 | • PII 자동 감지/마스킹 로직 <br> • 접근 로그 전수 기록 (SFR-010) <br> • RBAC으로 권한 최소화 <br> • 정기 보안 점검 (연 2회) |
| **외부 API 장애 (OpenAI, Upstage)** | 중간 | 낮음 | • 타임아웃 설정 (30초) <br> • 재시도 로직 (exponential backoff) <br> • 페일오버: OpenAI 실패시 Anthropic 대체 <br> • 로컬 모델(Ollama)을 백업으로 유지 |

**리스크 모니터링**
- 주간 스탠드업에서 리스크 상태 점검
- 일정 지연 발생시 즉시 우선순위 재조정
- 성능/비용 이슈는 MonitorPage 대시보드로 실시간 추적
