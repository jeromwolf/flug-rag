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
python tests/benchmark_kogas_law.py           # 한국가스공사법 벤치마크 (50문항)
python tests/benchmark_internal_rules.py      # 내부규정 벤치마크 (60문항)
python tests/benchmark_optimizer.py           # 자동 파라미터 최적화
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
├── frontend/             # React + TypeScript + MUI
│   └── src/
│       ├── pages/        # Chat, Admin, Documents, Monitor, QualityDashboard, AgentBuilder
│       └── components/   # ContentManager, CustomToolBuilder, EthicsPledge 등
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

### 한국가스공사법 (Phase 1 - 완료)
- 성공률: **100%** (50/50)
- 평균 신뢰도: 0.999
- 카테고리: factual(20), inference(15), multi_hop(10), negative(5)

### 내부규정 (Phase 2 - 진행 중)
- 성공률: **93.3%** (56/60)
- 평균 composite: 0.6495
- 카테고리: factual(25), inference(15), multi_hop(10), negative(10)
- 목표: 95%+ 성공률

## 테스트 계정

| 역할 | 사용자명 | 비밀번호 |
|------|---------|---------|
| Admin | admin | admin123 |
| Manager | manager | manager123 |
| User | user | user123 |

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

### Phase 2: 내부규정 벤치마크 튜닝 (진행 중)
- **현재 상태**: 60문항 93.3% 성공 (56/60), 평균 composite 0.6495
- **목표**: 95%+ 성공률
- **실패 문항 패턴**:
  - 답변 과잉 (factual Q3,Q22,Q24,Q25) - ROUGE 저하
  - 추론 부족 (inference Q27,Q39) - 규정 재진술만
  - 다중 규정 연결 실패 (multi_hop Q47,Q53)
  - 부정 질문 오답 (negative Q58,Q59) - 범위 외 규정 참조
- **벤치마크**: `python tests/benchmark_internal_rules.py`
- **골든 데이터셋**: `tests/golden_dataset_internal_rules.json`

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

### Phase 3: 나머지 데이터셋 (예정)
- 국외출장 결과보고서 (100 files)
- 인쇄홍보물 (7 files)
- ALIO 검색결과 (403 files)

## 데이터 위치

| 데이터셋 | 경로 | 파일 수 | 상태 |
|---------|------|--------|------|
| 한국가스공사법 | `data/sample_dataset/한국가스공사법/` | PDF + 관련법 | 벤치마크 완료 (100%) |
| 내부규정 | `data/uploads/한국가스기술공사_내부규정/` | 676 HWP | 인제스트 완료, 튜닝 중 |
| 국외출장 보고서 | `data/uploads/국외출장_결과보고서/` | 100 files | 인제스트 대기 |
| 인쇄홍보물 | `data/uploads/인쇄홍보물/` | 7 PDF | 인제스트 대기 |
| ALIO 검색결과 | `data/uploads/ALIO_한국가스기술공사_검색결과/` | 403 files | 인제스트 대기 |
