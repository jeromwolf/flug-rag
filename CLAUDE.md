# Flux RAG - 프로젝트 가이드

## 프로젝트 개요
한국가스기술공사 생성형 AI 플랫폼 - 종합 엔터프라이즈 AI 플랫폼 (RAG, Agent, MCP, 품질관리, 보안 컴플라이언스)

## 빠른 시작

```bash
# 백엔드 실행
cd backend
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 프론트엔드 실행
cd frontend
npm run dev
```

## 주요 명령어

### 서버 실행
- 백엔드: `uvicorn api.main:app --port 8000`
- 프론트엔드: `npm run dev` (포트 5173)
- Ollama: `ollama serve` (포트 11434)

### 테스트
```bash
cd backend
pytest tests/ -v                              # 전체 테스트
python tests/benchmark_all.py                 # 통합 벤치마크 (120문항, 4개 데이터셋)
python tests/benchmark_kogas_law.py           # 한국가스공사법 벤치마크 (50문항)
python tests/benchmark_internal_rules.py      # 내부규정 벤치마크 (60문항)
python tests/benchmark_parametric.py          # 자동 파라미터 최적화 (grid search)
python tests/failure_analyzer.py              # 실패 사례 자동 분석
```

### 문서 인제스트
```bash
python -c "
import asyncio
from pathlib import Path
from pipeline.ingest import IngestPipeline

async def main():
    pipeline = IngestPipeline()
    result = await pipeline.ingest(Path('data/sample.pdf'))
    print(f'{result.chunk_count} chunks')

asyncio.run(main())
"
```

## 환경 설정 (.env)

```bash
# LLM
DEFAULT_LLM_PROVIDER=ollama      # vllm, ollama, openai, anthropic
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Auth
AUTH_ENABLED=false               # true for production

# OCR
UPSTAGE_API_KEY=                 # Upstage Document Parse
OCR_PROVIDER=cloud               # cloud, onprem

# Advanced RAG
MULTI_QUERY_ENABLED=false
SELF_RAG_ENABLED=false
AGENTIC_RAG_ENABLED=false
QUERY_EXPANSION_ENABLED=false    # HyDE

# Sync
SYNC_ENABLED=false
SYNC_CRON=0 2 * * *             # 매일 새벽 2시
```

## 프로젝트 구조

```
flux-rag/
├── backend/
│   ├── api/              # FastAPI 라우트 (19개 모듈)
│   │   └── routes/       # chat, documents, sessions, admin, feedback, mcp,
│   │                     # workflows, quality, sync, folders, personal_knowledge,
│   │                     # ocr, ocr_training, statistics, logs, guardrails, content
│   ├── agent/            # Agent, MCP (10개 도구), Router, Builder
│   ├── auth/             # JWT, RBAC, LDAP, 윤리서약, 접근요청
│   ├── config/           # Pydantic 설정
│   ├── core/
│   │   ├── llm/          # LLM 프로바이더 (Ollama, OpenAI, Anthropic, vLLM) + 모델 레지스트리
│   │   ├── embeddings/   # 임베딩 (bge-m3) + 추적
│   │   ├── vectorstore/  # ChromaDB, Milvus + 분석기
│   │   ├── cache/        # Redis, 메모리 캐시
│   │   ├── db/           # AsyncSQLiteManager 베이스
│   │   └── performance/  # 배치 처리, 커넥션 풀
│   ├── pipeline/         # 문서 처리 파이프라인
│   │   ├── loaders/      # PDF, HWP, DOCX, XLSX, PPTX, TXT
│   │   └── ocr/          # OCR (Upstage Cloud/On-Prem)
│   ├── rag/              # RAG 체인, 검색, 품질, 가드레일
│   │   # chain, retriever, prompt, quality, evaluator,
│   │   # guardrails, golden_data, prompt_versioning, access_control,
│   │   # query_expander(HyDE), self_rag, agentic, multi_query,
│   │   # query_corrector, chunk_quality
│   ├── monitoring/       # Prometheus, 남용탐지, 알림
│   ├── scripts/          # 인제스트, 마이그레이션 스크립트
│   ├── data/
│   │   └── sample_dataset/한국가스공사법/  # 벤치마크 데이터
│   └── tests/            # pytest + 벤치마크
├── frontend/             # React 19 + TypeScript 5.9 + MUI v7 + Vite 7
│   └── src/
│       ├── pages/        # Chat, Admin, Documents, Monitor, QualityDashboard, AgentBuilder
│       ├── components/
│       │   └── chat/     # ChatSidebar, ChatTopBar, MessageBubble, SourcesPanel 등 8개 (ChatGPT 스타일)
│       ├── hooks/        # useStreamingChat, useSessions, useFeedback 등 5개
│       └── stores/       # Zustand 상태 관리 (appStore — session, model, temperature, darkMode)
├── k8s/                  # Kubernetes 배포
└── scripts/              # 유틸리티 스크립트
```

## 핵심 컴포넌트

### RAG 파이프라인
- **Retriever**: HybridRetriever (Vector + BM25)
- **Reranker**: FlashRank (ms-marco-MultiBERT-L-12)
- **Embedder**: BAAI/bge-m3 (1024 dim)
- **VectorStore**: ChromaDB (개발), Milvus (운영)

### LLM 프로바이더
- Ollama (로컬): qwen2.5:7b
- vLLM (운영): 고성능 서빙
- OpenAI/Anthropic (폴백)

### 고급 RAG 기법
- **Self-RAG**: 자기반성적 RAG, 환각 탐지 (`rag/self_rag.py`)
- **Multi-Query**: 다중 관점 쿼리 생성 (`rag/multi_query.py`)
- **Agentic RAG**: 동적 전략 라우팅 (`rag/agentic.py`)
- **HyDE**: 가설 문서 임베딩 기반 쿼리 확장 (`rag/query_expander.py`)
- **Query Corrector**: 오타/맞춤법 교정 (`rag/query_corrector.py`)

### 보안 및 컴플라이언스
- **가드레일**: 입출력 필터링, 프롬프트 인젝션 탐지 (`rag/guardrails.py`)
- **AI 윤리서약**: 사용자 윤리서약 관리 (`auth/ethics.py`)
- **접근요청**: 역할 변경 요청/승인 워크플로 (`auth/access_request.py`)
- **PII 탐지**: 한국형 개인정보 탐지 (주민번호, 전화번호 등) (`pipeline/pii_detector.py`)
- **남용탐지**: IP 블랙리스트, 이상 패턴 탐지 (`monitoring/abuse_detector.py`)

### 문서 관리
- **OCR**: Upstage Cloud/On-Prem 문서 파싱 (`pipeline/ocr/`)
- **문서 동기화**: 변경 감지 + 자동 재인제스트 (`pipeline/sync.py`, `pipeline/scheduler.py`)
- **재처리 큐**: 실패 문서 자동 재처리 (`pipeline/reprocess_queue.py`)
- **개인 지식공간**: 사용자별 문서 관리 (`api/routes/personal_knowledge.py`)
- **폴더 접근제어**: 폴더 기반 권한 관리 (`rag/access_control.py`)

### 품질 관리
- **청크 품질 분석**: 길이 분포, 중복, 의미 완결성 (`rag/chunk_quality.py`)
- **임베딩 추적**: 인제스트 작업 모니터링 (`core/embeddings/tracker.py`)
- **골든 데이터**: 전문가 검증 Q&A 관리 (`rag/golden_data.py`)
- **프롬프트 버저닝**: 버전 관리 + 롤백 (`rag/prompt_versioning.py`)

## 벤치마크 결과

### 통합 벤치마크 (4개 데이터셋, 120문항)
- 성공률: **95.8%** (115/120)
- 모델: qwen2.5:14b (Ollama, temperature 0.1)
- 벡터스토어: 39,739 chunks (ALIO 19,274 / 내부규정 19,035 / 홍보물 1,012 / 출장보고서 301 / 정관 89 / 법률 28)

| 데이터셋 | 문항 수 | 성공률 | 비고 |
|---------|--------|--------|------|
| 한국가스공사법 | 50 | **100%** (50/50) | Phase 1 완료 |
| 내부규정 | 60 | **98.3%** (59/60) | Phase 2 완료 |
| 인쇄홍보물 | 20 | **100%** (20/20) | Phase 3 완료 |
| ALIO 공시 | 20 | **100%** (20/20) | Phase 3 완료 |
| 국외출장 보고서 | 20 | **80%** (16/20) | PDF 텍스트 추출 품질 이슈 (OCR 재인제스트 필요) |

### 주요 튜닝 포인트
- 도메인별 시스템 프롬프트 자동 선택 (legal/technical/general)
- 카테고리별 few-shot 예시 (법률 8개, 일반 도메인 확장)
- 모델 크기 인식 프롬프팅 (7B/14B 별도 간결성 지시)
- 카테고리별 가중치 평가 (factual/inference/negative/multi_hop)
- 응답 검증 + 자동 재시도 (깨진 출력, 중국어 누출 감지)
- source_type 기반 자동 필터링 (질문 키워드 → ChromaDB 메타데이터 필터)

## 테스트 계정

| 역할 | 사용자명 | 비밀번호 |
|------|---------|---------|
| Admin | admin | admin123 |
| Manager | manager | manager123 |
| User | user | user123 |
| Viewer | viewer | viewer123 |

> **주의**: 기본 비밀번호 사용 시 최초 로그인 후 `POST /api/auth/change-password`로 변경 필요 (`must_change_password` 플래그).
> 비밀번호 요구사항: 최소 8자, 대문자 1자+, 숫자 1자+, 특수문자 1자+

## API 엔드포인트

### Core
- `POST /api/chat` - RAG 질의응답 (SSE 스트리밍)
- `POST /api/chat/stream` - SSE 스트리밍 응답
- `POST /api/documents/upload` - 문서 업로드
- `GET /api/documents` - 문서 목록
- `GET /api/sessions` - 대화 세션 목록
- `GET /health` - 헬스 체크

### Admin
- `GET /api/admin/system-info` - 시스템 정보
- `GET/PUT /api/admin/prompts` - 프롬프트 관리
- `GET/POST /api/admin/models` - 모델 레지스트리
- `GET /api/admin/prompt-versions` - 프롬프트 버전 관리

### Quality & Monitoring
- `GET /api/quality/*` - 품질 대시보드 (청크, 임베딩, 벡터 분포)
- `GET /api/statistics/*` - 사용 통계, 키워드, Excel 내보내기
- `GET /api/logs/*` - 접근 로그, 쿼리 이력

### Document Management
- `POST /api/sync/trigger` - 문서 동기화 트리거
- `GET /api/folders` - 지식 폴더 관리
- `POST /api/personal-knowledge/*` - 개인 지식공간
- `POST /api/ocr/process` - OCR 처리

### Security
- `GET/POST /api/guardrails/*` - 가드레일 규칙 관리
- `POST /api/auth/login` - 로그인 (JWT)
- `POST /api/auth/refresh` - 토큰 갱신
- `POST /api/auth/change-password` - 비밀번호 변경 (복잡성 검증 포함)

### Tools & Content
- `GET /api/mcp/tools` - MCP 도구 목록 (10개 내장 + 커스텀 도구 빌더)
- `POST /api/mcp/execute` - 도구 실행
- `GET/POST /api/content/*` - 공지사항, FAQ, 설문
- `GET /api/workflows/*` - 워크플로 프리셋

## 개발 가이드

### 새 LLM 프로바이더 추가
1. `core/llm/` 에 새 프로바이더 클래스 생성
2. `BaseLLM` 상속
3. `core/llm/__init__.py` 에 등록

### 새 문서 로더 추가
1. `pipeline/loaders/` 에 로더 클래스 생성
2. `BaseLoader` 상속
3. `pipeline/loader.py` 에 등록

## 트러블슈팅

### Ollama 연결 실패
```bash
# Ollama 상태 확인
curl http://localhost:11434/api/tags

# 모델 다운로드
ollama pull qwen2.5:7b
```

### ChromaDB 초기화
```bash
rm -rf data/chroma_db
# 문서 재인제스트
```

### 벡터스토어 문서 수 확인
```python
from core.vectorstore import create_vectorstore
import asyncio

async def check():
    vs = create_vectorstore()
    count = await vs.count()
    print(f"Total: {count} chunks")

asyncio.run(check())
```

## 개발 진행 상황

### Phase 0: P0 이슈 수정 (완료)
- **UserStore SQLite 마이그레이션**: `backend/auth/user_store.py` 전면 재작성. 인메모리 → SQLite(`data/users.db`). 5개 소비자 파일 async 전환 완료.
- **K8s Secret 보안**: `k8s/secret.yaml`을 `.gitignore`에 추가, `k8s/secret.yaml.example` 템플릿 생성

### Phase 1: 한국가스공사법 벤치마크 튜닝 (완료)
- **결과**: 50문항 100% 성공률, 평균 신뢰도 0.999
- **개선사항**:
  - 시스템 프롬프트 간결성 강화 (`prompts/system.yaml`)
  - 법률 도메인 few-shot 예시 6개로 확대 (`prompts/few_shot.yaml`)
  - 모델 크기 인식 프롬프팅 (`rag/prompt.py` - 7B 모델용 추가 간결성 지시)
  - 카테고리별 가중치 평가 (`rag/evaluator.py` - factual/inference/negative 별도 가중치)
  - 길이 패널티 도입 (2.5x 초과 답변 감점)

### Phase 2: 내부규정 벤치마크 튜닝 (완료)
- **결과**: 60문항 98.3% 성공 (59/60)
- **개선사항**:
  - 다중 규정 연계 처리 시스템 프롬프트 추가
  - 추론 거부 방지 지시 (규정 해석/의미 추론 허용)
  - few-shot 예시 8개로 확대 (factual + negative + inference + multi_hop 전 카테고리)
  - 산술 계산 명시적 허용 (변동액/차이/증감 계산)
  - 응답 검증 + 자동 재시도 로직 (깨진 출력 감지)
- **벤치마크**: `python tests/benchmark_internal_rules.py`
- **골든 데이터셋**: `tests/golden_dataset_internal_rules.json` (60문항)

### SFR 기능 개발 (완료)
- **SFR-002 사용자 컴플라이언스**: AI 윤리서약, 접근요청/승인 워크플로
- **SFR-003 안전 가드레일**: 입출력 필터링, 프롬프트 인젝션 탐지, 키워드/정규식 필터
- **SFR-005 문서 관리 고도화**: 동기화 엔진, PII 탐지, 개인 지식공간, 폴더 접근제어, 재처리 큐
- **SFR-008 품질 관리**: 청크 품질 분석, 임베딩 추적, 벡터 분포 분석, 품질 대시보드
- **SFR-009 콘텐츠/통계**: 사용 통계 + Excel 내보내기, 공지사항/FAQ/설문 관리
- **SFR-010 모니터링/보안**: 남용탐지, 리소스 알림, 로그 검색
- **SFR-014 모델/프롬프트**: LLM 모델 레지스트리, 프롬프트 버저닝 + 롤백
- **SFR-015 OCR**: Upstage Cloud/On-Prem OCR, 학습 데이터 수집
- **SFR-017 골든 데이터**: 전문가 검증 Q&A 관리 + 평가
- **SFR-018 커스텀 도구**: 노코드 MCP 도구 빌더, 규정검토/안전점검 도메인 도구
- **고급 RAG**: Self-RAG, Multi-Query, Agentic RAG, HyDE, Query Corrector

### Phase 3: 나머지 데이터셋 벤치마크 (완료)
- **인쇄홍보물**: 20문항 100% 성공, 골든 데이터셋 `tests/golden_dataset_brochure.json`
- **ALIO 공시**: 20문항 100% 성공, 골든 데이터셋 `tests/golden_dataset_alio.json`
- **국외출장 보고서**: 20문항 80% 성공, 골든 데이터셋 `tests/golden_dataset_travel.json`
  - 잔여 실패 4건: PDF 텍스트 추출 품질 이슈 (Upstage OCR 재인제스트 필요)

### 보안/성능 코드리뷰 수정 (2025-02 완료)

총 14개 파일 수정, Architect 3라운드 리뷰 전부 APPROVED.

#### P0 보안 수정 (2건)
- **JWT 토큰 타입 검증**: `auth/jwt_handler.py` — `verify_token()`에 `required_type` 파라미터 추가. Refresh 토큰을 Access로 악용하는 취약점 차단
- **기본 비밀번호 변경 강제**: `auth/user_store.py` — `must_change_password` 컬럼 추가, `change_password()` 메서드 (동일 비밀번호 재사용 차단). `auth/routes.py` — `POST /auth/change-password` 엔드포인트 + 복잡성 검증 (대문자+숫자+특수문자). `auth/audit.py` — `PASSWORD_CHANGE` 감사 이벤트 추가
- 수정 파일: `auth/jwt_handler.py`, `auth/routes.py`, `auth/user_store.py`, `auth/audit.py`, `api/main.py`, `tests/test_auth.py`

#### P1 정확성 수정 (3건)
- **스트리밍 파이프라인 품질 복원**: `rag/chain.py` — `stream_query()`에 누락된 5개 전처리 단계 추가 (query correction, terminology, agentic routing, multi-hop, source_type 필터)
- **Temperature 기본값 통일**: `core/llm/factory.py` — 하드코딩 0.7 제거, `settings.llm_temperature` 참조로 변경
- **입력 길이 제한 통일**: `rag/guardrails.py` — `MAX_INPUT_LENGTH` 5000 → 10000 (API 스키마와 일치)

#### P2 성능 수정 (4건)
- **DB 커넥션 최적화**: `core/db/base.py` — 중복 PRAGMA 제거. `agent/memory.py` — AsyncSQLiteManager 상속으로 전면 재작성. `auth/user_store.py` — 12개 raw connect → `get_connection()` 통합
- **배치 인제스트 병렬화**: `pipeline/ingest.py` — `asyncio.Semaphore(5)` + `gather()` 동시 처리
- **BM25 이벤트루프 블로킹 해소**: `rag/retriever.py` — `asyncio.to_thread()` 오프로딩
- **AbuseDetector 영속화**: `monitoring/abuse_detector.py` — AsyncSQLiteManager 기반 SQLite 재작성 (블랙리스트/이벤트 영속, 레이트리밋 인메모리 유지)

#### P3 잔여 이슈 (미수정, 향후 작업)
- 키워드 하드코딩 → YAML 외부화 (`rag/chain.py` source_type 키워드 맵)
- 싱글톤 패턴 8종 통일 (일부 async, 일부 sync, 일부 module-level)
- 프론트엔드 테스트 커버리지 0% → 최소 단위 테스트 추가 필요
- DB 마이그레이션 시스템 부재 → Alembic 또는 자체 마이그레이션 도입 필요

### 프론트엔드 UI 고도화 (2026-02 완료)

ChatGPT/Claude.ai 수준의 모던 AI 챗봇 UI로 전면 리디자인. 11개 파일 수정.

#### 테마 및 기반
- **컬러 시스템**: primary `#10a37f` (ChatGPT 그린), secondary `#6e6e80`
- **다크모드**: bg `#212121`/`#2f2f2f`, 라이트: `#ffffff`/`#f7f7f8`
- **타이포**: Pretendard Variable 폰트, body1 0.9375rem/1.7
- **글로벌 애니메이션**: fadeInUp keyframes, thin 스크롤바

#### 사이드바 (항상 다크)
- bg `#171717`, 텍스트 `#ececec`/`#8e8ea0`, 너비 260px
- "Flux AI" 브랜딩 + AutoAwesome 아이콘

#### 메시지 버블 → 아바타+텍스트 레이아웃
- 중앙 정렬 maxWidth 768px, 사용자(보라 #5436DA)/AI(그린) 아바타
- 코드 블록: 언어 헤더바 + 복사 버튼, JetBrains Mono
- React.memo 적용, fadeInUp 애니메이션
- hover 시 액션 버튼 노출 (opacity 0→1)

#### 입력바
- 중앙 정렬 라운드 border, react-dropzone 파일 첨부
- ArrowUpward 원형 전송 버튼, 5000자 초과 시 카운터 표시

#### 톱바
- 미니멀: [Menu] ... [Model Selector] ... [Copy] [DarkMode] [Settings]
- `adminApi.listModels()` API 연동 (폴백 하드코딩)
- 대화 클립보드 복사 버튼

#### 소스 패널
- Accordion → 카드 그리드 (flex-wrap), 점수 Chip 컬러 코딩

#### 신규 기능
- **사용자 메시지 편집+재전송**: 해당 메시지 이후 삭제 → 입력창 복원
- **파일 첨부 UI**: Chip 프리뷰 (백엔드 업로드 미지원, UI only)
- **대화 복사**: 전체 대화 텍스트 클립보드 복사

#### 수정 파일 (11개)
`index.html`, `App.tsx`, `appStore.ts`, `ChatSidebar.tsx`, `ChatTopBar.tsx`, `ChatMessageList.tsx`, `MessageBubble.tsx`, `ChatInputBar.tsx`, `SourcesPanel.tsx`, `useStreamingChat.ts`, `ChatPage.tsx`

### 잔여 작업
- **출장보고서 OCR 재인제스트**: 깨진 PDF 텍스트 수정 (Upstage Document Parse 적용)
- **운영 배포 준비**: vLLM 서빙, K8s 매니페스트, Redis 캐시
- **배포 플랫폼**: RunPod A40 Community Cloud ($0.35/hr, 월 ~$259) 권장 — GCP 대비 3~5배 저렴

## 데이터 위치

| 데이터셋 | 경로 | 파일 수 | 청크 수 | 벤치마크 |
|---------|------|--------|--------|---------|
| 한국가스공사법 | `data/sample_dataset/한국가스공사법/` | PDF + 관련법 | 28 (법률) + 89 (정관) | 100% (50/50) |
| 내부규정 | `data/uploads/한국가스기술공사_내부규정/` | 676 HWP | 19,035 | 98.3% (59/60) |
| 국외출장 보고서 | `data/uploads/국외출장_결과보고서/` | 100 files | 301 | 80% (16/20) |
| 인쇄홍보물 | `data/uploads/인쇄홍보물/` | 7 PDF | 1,012 | 100% (20/20) |
| ALIO 검색결과 | `data/uploads/ALIO_한국가스기술공사_검색결과/` | 403 files | 19,274 | 100% (20/20) |
