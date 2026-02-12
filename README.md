# Flux RAG

한국가스기술공사 생성형 AI 플랫폼 - 엔터프라이즈 RAG 프레임워크

## 개요

Flux RAG는 기업 문서 기반 질의응답 시스템을 위한 완전한 RAG(Retrieval-Augmented Generation) 프레임워크입니다. 하이브리드 검색, 고급 RAG 기법, 문서 관리, 보안 가드레일, 품질 관리까지 운영 환경에 필요한 전체 스택을 제공합니다.

### 주요 기능

- **하이브리드 검색**: 벡터 유사도 + BM25 키워드 검색 + FlashRank 리랭킹
- **고급 RAG**: Self-RAG, Multi-Query, Agentic RAG, HyDE 쿼리 확장
- **다중 LLM**: vLLM, Ollama, OpenAI, Anthropic
- **다중 VectorDB**: ChromaDB (개발), Milvus (운영)
- **문서 파이프라인**: PDF, HWP, DOCX, XLSX, PPTX + OCR (Upstage)
- **보안**: JWT + RBAC 4단계 + LDAP + 가드레일 + PII 탐지
- **품질 관리**: 청크 분석, 임베딩 추적, 골든 데이터, 벤치마크
- **Agent System**: 10개 MCP 도구 + 커스텀 도구 빌더 + 워크플로
- **모니터링**: Prometheus + Grafana + 남용탐지 + 알림

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| Backend | Python 3.11+, FastAPI, asyncio |
| Frontend | React 18, TypeScript, MUI, Recharts |
| LLM | vLLM, Ollama, OpenAI, Anthropic |
| Embedding | BAAI/bge-m3 (1024 dim) |
| Reranker | FlashRank (ms-marco-MultiBERT-L-12) |
| Vector DB | ChromaDB, Milvus |
| Database | SQLite (개발), PostgreSQL (운영) |
| OCR | Upstage Document Parse (Cloud/On-Prem) |
| Cache | Redis (선택적) |
| Monitoring | Prometheus + Grafana |
| Container | Docker, Kubernetes |

## 빠른 시작

### 1. 백엔드 설치

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 필요한 설정 수정
```

### 2. 문서 인제스트

```bash
cd backend

# 샘플 문서 인제스트
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

# 내부규정 일괄 인제스트
python scripts/ingest_internal_rules.py
```

### 3. 서버 실행

```bash
# 백엔드 (포트 8000)
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 프론트엔드 (포트 5173)
cd frontend
npm install && npm run dev
```

### 4. 접속

- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

**테스트 계정:**
| 역할 | 사용자명 | 비밀번호 |
|------|---------|---------|
| Admin | admin | admin123 |
| Manager | manager | manager123 |
| User | user | user123 |
| Viewer | viewer | viewer123 |

## 아키텍처

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  Frontend    │     │  Backend (FastAPI)                    │
│  React+MUI   │────▶│                                      │
│  6 Pages     │     │  ┌─────────┐  ┌──────────────────┐  │
└─────────────┘     │  │ Auth    │  │ API Routes (19)  │  │
                    │  │ JWT+RBAC│  │ chat, docs, admin │  │
                    │  └─────────┘  │ quality, sync ... │  │
                    │               └────────┬─────────┘  │
                    │                        │             │
                    │  ┌─────────────────────▼──────────┐  │
                    │  │     RAG Engine                  │  │
                    │  │  Guardrails → Query Correct     │  │
                    │  │  → HyDE/MultiQ → Hybrid Search │  │
                    │  │  → Rerank → LLM → Self-RAG     │  │
                    │  └──────┬──────────────┬──────────┘  │
                    │         │              │             │
                    │  ┌──────▼─────┐ ┌─────▼──────────┐  │
                    │  │ VectorDB   │ │ LLM Provider   │  │
                    │  │ Chroma/    │ │ Ollama/vLLM/   │  │
                    │  │ Milvus     │ │ OpenAI/Claude  │  │
                    │  └────────────┘ └────────────────┘  │
                    │                                      │
                    │  ┌────────────┐ ┌────────────────┐  │
                    │  │ Pipeline   │ │ Agent/MCP      │  │
                    │  │ Ingest/OCR │ │ 10 Tools +     │  │
                    │  │ PII/Sync   │ │ Custom Builder │  │
                    │  └────────────┘ └────────────────┘  │
                    └──────────────────────────────────────┘
```

## 프로젝트 구조

```
flux-rag/
├── backend/
│   ├── api/                  # FastAPI 라우트 (19개 모듈)
│   │   └── routes/           # chat, documents, sessions, admin, feedback,
│   │                         # mcp, workflows, quality, sync, folders,
│   │                         # personal_knowledge, ocr, statistics, logs,
│   │                         # guardrails, content, ocr_training
│   ├── agent/                # Agent 시스템
│   │   ├── mcp/              # MCP 도구 (10개 내장 + 커스텀)
│   │   ├── builder/          # Agent 빌더 + 프리셋
│   │   ├── collaboration/    # 다중 에이전트 협업
│   │   ├── chaining/         # 에이전트 체이닝
│   │   └── monitor/          # 실행 모니터링
│   ├── auth/                 # 인증/권한
│   │   # routes, user_store, jwt, ldap, rbac,
│   │   # ethics(윤리서약), access_request(접근요청)
│   ├── config/               # Pydantic 설정
│   ├── core/
│   │   ├── llm/              # LLM 프로바이더 + 모델 레지스트리
│   │   ├── embeddings/       # bge-m3 임베딩 + 추적
│   │   ├── vectorstore/      # ChromaDB, Milvus + 분석기
│   │   ├── cache/            # Redis, 메모리 캐시
│   │   ├── db/               # AsyncSQLiteManager 베이스
│   │   └── performance/      # 배치 처리, 커넥션 풀, 프로파일링
│   ├── pipeline/             # 문서 처리 파이프라인
│   │   ├── loaders/          # PDF, HWP, DOCX, XLSX, PPTX, TXT
│   │   └── ocr/              # Upstage Cloud/On-Prem OCR
│   │   # ingest, chunker, metadata, pii_detector,
│   │   # sync, scheduler, document_monitor, reprocess_queue
│   ├── rag/                  # RAG 엔진
│   │   # chain, retriever, prompt, quality, evaluator,
│   │   # guardrails, self_rag, multi_query, agentic,
│   │   # query_expander(HyDE), query_corrector,
│   │   # golden_data, prompt_versioning, access_control, chunk_quality
│   ├── monitoring/           # Prometheus, 남용탐지, 알림
│   ├── scripts/              # 인제스트/마이그레이션 스크립트
│   ├── data/                 # 데이터 + 벤치마크 결과
│   └── tests/                # pytest + 벤치마크 (18개 테스트 파일)
├── frontend/
│   └── src/
│       ├── pages/            # Chat, Admin, Documents, Monitor,
│       │                     # QualityDashboard, AgentBuilder
│       ├── components/       # ContentManager, CustomToolBuilder,
│       │                     # EthicsPledge, AccessRequest 등
│       ├── api/              # API 클라이언트 (15개 서비스 모듈)
│       └── contexts/         # Auth Context
├── k8s/                      # Kubernetes 매니페스트
└── scripts/                  # 유틸리티 스크립트
```

## API 엔드포인트 (19개 라우트 모듈)

### Core
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/chat` | RAG 질의응답 (SSE 스트리밍) |
| POST | `/api/documents/upload` | 문서 업로드 |
| GET | `/api/documents` | 문서 목록 |
| GET | `/api/sessions` | 대화 세션 목록 |
| GET | `/health` | 헬스 체크 |

### Admin & Management
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/admin/system-info` | 시스템 정보 |
| GET/PUT | `/api/admin/prompts` | 프롬프트 관리 |
| GET/POST | `/api/admin/models` | 모델 레지스트리 |
| GET/POST | `/api/guardrails/rules` | 가드레일 규칙 관리 |
| GET/POST | `/api/content/*` | 공지사항, FAQ, 설문 |

### Quality & Monitoring
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/quality/*` | 품질 대시보드 |
| GET | `/api/statistics/*` | 사용 통계 + Excel 내보내기 |
| GET | `/api/logs/*` | 접근 로그, 쿼리 이력 |
| GET | `/api/feedback` | 사용자 피드백 |

### Document Management
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/sync/trigger` | 문서 동기화 |
| GET/POST | `/api/folders` | 지식 폴더 관리 |
| POST | `/api/personal-knowledge/*` | 개인 지식공간 |
| POST | `/api/ocr/process` | OCR 처리 |

### Auth & Tools
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/auth/login` | 로그인 (JWT) |
| POST | `/api/auth/refresh` | 토큰 갱신 |
| GET | `/api/mcp/tools` | MCP 도구 목록 |
| POST | `/api/mcp/execute` | 도구 실행 |
| GET | `/api/workflows` | 워크플로 프리셋 |

## 벤치마크 결과

### 한국가스공사법 (50문항) - 완료
| 항목 | 결과 |
|------|------|
| 성공률 | **100%** (50/50) |
| 평균 신뢰도 | 0.999 |
| 카테고리 | factual(20), inference(15), multi_hop(10), negative(5) |

### 내부규정 (60문항) - 진행 중
| 항목 | 결과 |
|------|------|
| 성공률 | **93.3%** (56/60) |
| 평균 composite | 0.6495 |
| 카테고리 | factual(25), inference(15), multi_hop(10), negative(10) |
| 목표 | 95%+ 성공률 |

## 환경 변수

```bash
# LLM
DEFAULT_LLM_PROVIDER=ollama       # vllm, ollama, openai, anthropic
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Auth
AUTH_ENABLED=false                # true for production
JWT_SECRET_KEY=your-secret-key

# OCR
UPSTAGE_API_KEY=                  # Upstage Document Parse
OCR_PROVIDER=cloud                # cloud, onprem

# Advanced RAG
MULTI_QUERY_ENABLED=false
SELF_RAG_ENABLED=false
AGENTIC_RAG_ENABLED=false
QUERY_EXPANSION_ENABLED=false     # HyDE

# Sync
SYNC_ENABLED=false
SYNC_CRON=0 2 * * *              # 매일 새벽 2시

# Optional
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=false
PROMETHEUS_ENABLED=false
```

## 테스트

```bash
cd backend

# 단위 테스트
pytest tests/ -v

# 벤치마크
python tests/benchmark_kogas_law.py         # 한국가스공사법 (50문항)
python tests/benchmark_internal_rules.py    # 내부규정 (60문항)
python tests/benchmark_optimizer.py         # 자동 파라미터 최적화

# 커버리지
pytest tests/ --cov=. --cov-report=html
```

## Docker 배포

```bash
# 전체 스택 실행
docker-compose up -d

# Kubernetes 배포
kubectl apply -f k8s/

# OCR On-Prem (GPU)
cd backend && docker-compose -f docker-compose.ocr.yml up -d
```

## 데이터셋

| 데이터셋 | 파일 수 | 상태 |
|---------|--------|------|
| 한국가스공사법 | PDF + 관련법 | 벤치마크 완료 (100%) |
| 내부규정 | 676 HWP | 인제스트 완료, 벤치마크 93.3% |
| 국외출장 보고서 | 100 files | 인제스트 대기 |
| 인쇄홍보물 | 7 PDF | 인제스트 대기 |
| ALIO 검색결과 | 403 files | 인제스트 대기 |

## 라이선스

MIT License
