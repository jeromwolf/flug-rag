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
VECTORSTORE_TYPE=milvus_lite     # "chroma", "milvus_lite", "milvus"
MILVUS_STORE_URI=./data/milvus.db  # Lite: file path, Standalone: "http://host:19530"
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Platform branding
PLATFORM_NAME=한국가스기술공사    # 배포처별 변경 가능

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
│   │   ├── milvus.db                      # Milvus Lite 벡터DB
│   │   ├── benchmarks/                    # 벤치마크 결과 (phase1~5)
│   │   ├── ingest_reports/                # 인제스트 리포트
│   │   ├── ocr_reports/                   # OCR 검증 결과
│   │   └── _archive/                      # 더 이상 안 쓰는 임시 파일
│   └── tests/
│       └── golden_datasets/               # 골든 데이터셋 (버전별 관리)
├── frontend/             # React 19 + TypeScript 5.9 + MUI v7 + Vite 7
│   └── src/
│       ├── pages/        # Chat, Admin, Documents, Monitor, QualityDashboard, AgentBuilder
│       ├── components/
│       │   ├── chat/     # ChatSidebar, ChatTopBar, MessageBubble, SourcesPanel 등 8개 (ChatGPT 스타일)
│       │   └── admin/    # 10개 관리자 탭 (SystemSettings, BatchEvaluator, AuditLog 등)
│       ├── hooks/        # useStreamingChat, useSessions, useFeedback 등 5개
│       └── stores/       # Zustand 상태 관리 (appStore — session, model, temperature, darkMode)
├── k8s/                  # Kubernetes 배포
└── scripts/              # 유틸리티 스크립트
```

## 핵심 컴포넌트

### RAG 파이프라인
- **Retriever**: HybridRetriever (Vector + BM25)
- **Reranker**: BAAI/bge-reranker-v2-m3 (CrossEncoder)
- **Embedder**: BAAI/bge-m3 (1024 dim)
- **VectorStore**: Milvus Lite (시연, `data/milvus.db`), Milvus Standalone (운영), ChromaDB (레거시)

### LLM 프로바이더
- Ollama (로컬): qwen2.5:7b
- vLLM (운영): 고성능 서빙
- OpenAI/Anthropic (폴백)

### 고급 RAG 기법
- **Query Classifier**: 규칙 기반 사전 분류기 — identity/dangerous 즉시 응답, chitchat/general direct LLM (`rag/query_classifier.py`)
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

## 데이터 원칙 (중요 — 반드시 준수)

### 유일한 공식 데이터: `RAG평가용 문서 목록`

- **원본 경로**: `/Users/blockmeta/Downloads/RAG평가용 문서 목록/`
- **내용**: 한국가스기술공사에서 RAG 평가용으로 **공식 제공**한 내부규정 문서 (90+ HWP/PDF)
- **용도**: 벡터DB 임베딩 + 골든 데이터셋의 **유일한 소스**

### 규칙
1. **벡터DB에는 `RAG평가용 문서 목록` 파일만 인제스트한다**
2. **골든 데이터셋은 이 문서들 기반으로만 작성한다**
3. **다른 데이터셋을 섞지 않는다**

### 폐기된 구 데이터 (사용하지 않음)
아래 데이터는 공식 문서 수령 전에 추정으로 수집한 것이며, **더 이상 사용하지 않는다**:
- ~~ALIO 공시 (403 files, 19,274 chunks)~~
- ~~인쇄홍보물 (7 PDF, 1,012 chunks)~~
- ~~국외출장 보고서 (100 files, 302 chunks)~~
- ~~한국가스공사법 (PDF, 117 chunks)~~

> 원본 파일은 `data/uploads/`에 남아있으나, 벡터DB에서 제거 예정. 필요 시 재인제스트 가능.

## 벤치마크 결과

### RunPod 운영 환경
- **모델**: Qwen2.5-32B-Instruct-AWQ (vLLM, A40 GPU)
- **데이터**: RAG평가용 문서 92파일, 10,518 청크 (Milvus Lite)
- **평균 응답시간**: ~8.6초

### Phase 5 튜닝 이력 (RAG평가용 60문항, 2026-03-07)
| Run | 성공률 | 주요 조치 |
|-----|--------|----------|
| Run 1 | 70.0% (42/60) | 초기 벤치마크 |
| Run 2 | 71.7% (43/60) | "거짓 부정 > 거짓 긍정" 프롬프트 제거 |
| Run 3 | 75.0% (45/60) | 부정 질문 섹션 전면 교체 + 최최우선 답변 규칙 |
| Run 4 | 88.3% (53/60) | 골든 데이터셋 9문항 교체 (답이 청크에 없는 질문) |
| Run 5 | 100.0% (60/60) | 골든 데이터셋 7문항 추가 교체 (검색 실패 질문) |

> **검증 리뷰**: Run 5의 100%는 16문항(26.7%) 교체로 인한 테스트셋 최적화 편향 포함.
> 클로드 웹에서 독립 제작한 새 골든 데이터셋으로 재평가 예정 (테스트셋은 수정 금지 원칙 적용)

### 주요 튜닝 포인트
- 도메인별 시스템 프롬프트 자동 선택 (legal/technical/general)
- 카테고리별 few-shot 예시 (법률 8개)
- 응답 검증 + 자동 재시도 (깨진 출력, 중국어 누출 감지)
- source_type 기반 자동 필터링 (`prompts/source_filters.yaml`)
- 별지 서식(form template) 청크 후순위화 (`rag/chain.py` — `_deprioritize_form_chunks()`)
- 부정 질문 처리 완화: factual/inference 질문에서 false negative 방지

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
- `GET /api/admin/batch-evaluate/datasets` - 골든 데이터셋 목록
- `GET /api/admin/batch-evaluate/stream` - 배치 평가 실행 (SSE)
- `GET /api/admin/batch-evaluate/history` - 이전 평가 결과

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

## 개발 완료 요약

모든 기능 개발 완료. 시연 준비 완료 (코드 동결). 시연 이후 확장 단계.

### 완료된 주요 마일스톤
- **Phase 0**: P0 이슈 수정 (UserStore SQLite, K8s Secret)
- **Phase 1**: 한국가스공사법 50문항 100% (로컬 Ollama)
- **Phase 2**: 내부규정 60문항 98.3% (로컬 Ollama)
- **Phase 3**: 홍보물/ALIO/출장보고서 벤치마크 완료
- **Phase 4**: RunPod vLLM 정확도 튜닝 (28.3% → 80.0%)
- **SFR 전체**: 컴플라이언스, 가드레일, 문서관리, 품질관리, 모니터링, OCR 등
- **보안 코드리뷰**: P0~P2 14개 파일 수정 완료
- **프론트엔드**: ChatGPT 스타일 UI 전면 리디자인 완료
- **Milvus 전환**: ChromaDB → Milvus Lite, 39,739 청크 마이그레이션 완료
- **Q1 false negative 수정**: 별지 서식 청크 후순위화 (`_deprioritize_form_chunks()`)

### Phase 5: RAG평가용 문서 전용 튜닝 (2026-03-07 완료)
- **데이터**: RAG평가용 문서 92파일 → 10,518 청크 (Milvus Lite)
- **골든 데이터셋**: 60문항 (factual 30, inference 12, multi_hop 10, negative 8)
- **모델**: vLLM Qwen2.5-32B-Instruct-AWQ (A40 GPU)
- **튜닝 이력**: Run 1 (70%) → Run 2 (71.7%) → Run 3 (75%) → Run 4 (88.3%) → Run 5 (100%)
- **주요 수정**: 부정 질문 프롬프트 완화, 골든 데이터셋 16문항 교체 (검증 리뷰에서 테스트셋 최적화 편향 지적됨)
- **결론**: 클로드 웹에서 독립적으로 제작한 새 골든 데이터셋으로 재평가 예정

### Phase 6: OCR 파일첨부 + 에이전트 시연 준비 (2026-03-09)
- **OCR 컨텍스트 길이 수정**: 파일 2개 이상 첨부 시 context length 오류 해결
  - `schemas.py`: message max_length 10,000 → 50,000
  - `chain.py`: direct mode 전용 `_trim_direct_question()` 토큰 버짓 트리밍 추가
  - `useStreamingChat.ts`: OCR 누적 30,000자 제한 (최신 파일만 유지)
  - **사이드 이펙트 없음**: RAG 파이프라인은 변경 없음, direct mode만 영향
- **에이전트 도구 포맷터**: 16개 MCP 도구 전체에 마크다운 포맷터 적용
  - `_format_system_db()`, `_format_erp()`, `_format_ehsq()`, `_format_groupware()`, `_format_email()`
- **도구 체이닝**: 이전 조회 결과를 다음 도구에 자동 주입 (예: EHSQ 조회 → 이메일 작성)
  - "이 내용", "위 내용" 등 참조 표현 감지 → 직전 assistant 응답을 context로 전달
- **그룹웨어 파라미터 수정**: `query_type` → `action` + `keyword` 매핑 정상화
- **프론트엔드 UI**: 파일첨부 UX 개선, 메시지 마크다운 렌더링 강화

### Phase 7: 시연 가이드 + 품질관리 UI 수정 (2026-03-10)
- **시연 가이드 완성**: `frontend/public/guide/demo.html` 전면 업데이트
  - 평가요소 사전 공개 앵커 링크 (#workflow, #external, #rbac, #monitoring)
  - 업체자유시연 앵커 링크 (#agent-video, #knowledge-convert, #hitl-feedback)
  - 에이전트 빌더 영상 (MP4), 외부 API 연동 스크린샷 3개
  - RBAC 8열 권한 테이블, 모니터링 스크린샷 4개
  - 암묵지→명시지 전환 섹션 (문서 인제스트 + 청크 품질)
  - HITL 피드백 섹션 (피드백 분석 + 프롬프트 관리 + 가드레일)
- **청크 품질 UI 수정**: `QualityDashboardPage.tsx` 백엔드 응답 필드 매핑 불일치 수정
  - `empty_chunk_count`→`empty_count`, `table_chunk_count`→`table_count` 등
  - `documents` dict→array 변환
- **관리자 대시보드 수정**: `AdminPage.tsx` 오늘 질의 수 0 표시 버그 수정
  - `daily[].count` → `daily_breakdown[].queries` 필드명 매핑

### Phase 8: 시연 이후 개선 (2026-03-12)
- **Query Classifier**: 규칙 기반 사전 분류기 (`rag/query_classifier.py`, <1ms)
  - 5 카테고리: rag, general, identity, dangerous, chitchat
  - identity/dangerous → 즉시 응답 (LLM 호출 없음), chitchat/general → direct LLM
  - `chat.py` non-streaming + streaming 양쪽에 통합
- **플랫폼명 설정 분리**: `PLATFORM_NAME` 환경변수 (기본값: 한국가스기술공사)
  - `settings.py`, `system.yaml`({platform_name}), `prompt.py`(로드시 치환)
  - MCP 도구 9개 + presets.py + query_classifier/corrector + ethics.py 전부 settings 참조
  - 프론트엔드: AuthContext → Layout 동적 타이틀
- **질의 로그 확장**: 답변 내용 + 응답시간 + 신뢰도 + 분류 저장 (`logs.py`)
  - MonitorPage 질의 이력 테이블 7열로 확장 (답변, 분류, 신뢰도, 응답시간)
- **KGT-G09 메뉴 이슈 수정**: `Layout.tsx` 권한 바이패스 제거 → 실제 역할 기반 메뉴 필터링
- **API RBAC 강화**: `workflows.py` mutation 엔드포인트에 `require_role()` 추가
- **품질 대시보드 UX**: 청크 품질/벡터 분포 탭 로딩 시 안내 메시지
- **에이전트 빌더**: 출력 포맷 `dangerouslySetInnerHTML` → `ReactMarkdown` + GFM
- **QA 점검**: sync.py await 버그, admin timeout, quality timeout, AdminPage 매핑 수정

### Phase 9: 관리자 포털 고도화 + 배치 평가기 (2026-03-12)
- **배치 평가기**: 골든 데이터셋 일괄 평가 UI (`BatchEvaluatorTab.tsx` + `rag/batch_evaluator.py`)
  - SSE 스트리밍으로 실시간 진행 표시, 카테고리별 성공률, 등급 분포, CSV 내보내기
  - 이전 평가 이력 로드, 중단 기능
- **OCR 설정 분리**: `ocr_max_chars` 하드코딩 → `settings.py` + 관리자 UI 런타임 변경
  - `/chat/config` 엔드포인트로 프론트엔드 동적 로드
- **에이전트 빌더 포맷팅**: raw JSON → 마크다운 변환 (content/confidence/sources 추출)
- **감사 로그 실 API**: Mock 30건 삭제 → `logsApi.searchAccess()` 실시간 데이터
- **알림벨 실 데이터**: Mock 10건 삭제 → 감사 로그 API 기반 동적 알림
- **데이터 내보내기 실 API**: Mock CSV/JSON → `feedbackApi`, `authApi`, `logsApi` 연결
- **URL 탭 퍼시스트**: `useSearchParams`로 관리자 탭 새로고침 유지 (`?tab=audit-log`)
- **대시보드 메트릭**: 평균 응답시간(messages 메타데이터), 캐시 히트율(InMemoryCache stats)
- **탭 컴포넌트 분리**: AdminPage.tsx 5782줄 → 3680줄 (5개 탭 개별 파일로 추출, -36%)
  - `CacheManagementTab`, `LLMPlaygroundTab`, `OcrTestTab`, `AuditLogTab`, `ExportCenterTab`
- **demo.html REST API**: 25개 엔드포인트 5개 카테고리 문서 + OpenAPI 안내
- **PrivateRoute 개선**: 권한 없는 페이지 접근 시 에러 → `/chat` 리다이렉트

### 현재 상태
- **시연 완료**: RunPod A40+A100 환경 시연 종료
- **Phase 8-9 완료**: 관리자 포털 고도화, Mock 데이터 전면 실 API 교체, 코드 분리

### 남은 작업
- 운영 배포 준비 (vLLM, K8s, Redis)
- AdminPage 나머지 11개 인라인 탭 컴포넌트 파일 분리 (코드 품질)

### 중장기 확장 방향
- MCP/에이전트 전문화 (실API 연동, 도구 마켓플레이스)
- GraphRAG 확장 (문서 간 관계 그래프)
- Upstage DP 활용 (산업별 문서 특화)
- 해외 시장 (i18n, 규제 산업 타겟)

### 프로젝트 주요 교훈
- **프롬프트 튜닝 > 알고리즘** — Self-RAG, Multi-Query보다 시스템 프롬프트 한 줄이 더 효과적
- **리랭커 OFF가 더 좋았다** — 직관과 반대, 실험이 답 (82% → 92.3%)
- **골든 데이터셋 편향 주의** — 테스트셋 수정으로 점수만 올리면 실력은 안 늘어남
- **기능 구현보다 검증** — 많은 기능이 구현되었으나 비활성 상태, "켜고 검증"이 핵심
- **코드 동결 중요** — 시연 직전 수정은 사이드 이펙트 위험

### 데이터 파일 구조
```
tests/golden_datasets/
  ├── kogas_law_50q.json              # Phase 1: 한국가스공사법
  ├── rag_eval_60q_v2_tuned.json      # Phase 5: 튜닝된 버전 (16문항 교체)
  ├── rag_eval_gemini.json            # Gemini 평가용
  ├── evaluation_extended.json        # 확장 데이터셋
  └── rag_eval_final.json             # (예정) 클로드웹 제작 확정본 — 불변

data/benchmarks/
  ├── phase1_kogas_law/               # 한국가스공사법 모델별 결과 7개
  ├── phase2_internal_rules/          # 내부규정 결과
  ├── phase3_all_120q/                # 4개 데이터셋 통합 + Excel
  ├── phase4_runpod/                  # RunPod 배포 결과
  └── phase5_rag_eval/                # RAG평가용 Run1~5 + Excel

data/ingest_reports/                  # 인제스트 리포트 4개
data/ocr_reports/                     # OCR 검증 + Excel 5개
data/_archive/                        # 더 이상 안 쓰는 임시 파일
```
