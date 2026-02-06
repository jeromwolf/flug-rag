# Flux RAG

한국가스기술공사 생성형 AI 플랫폼 - RAG 프레임워크

## 개요

Flux RAG는 기업 문서 기반 질의응답 시스템을 위한 완전한 RAG(Retrieval-Augmented Generation) 프레임워크입니다.

### 주요 기능

- **하이브리드 검색**: 벡터 유사도 + BM25 키워드 검색
- **Reranking**: FlashRank 기반 검색 결과 재정렬
- **다중 LLM 지원**: vLLM, Ollama, OpenAI, Anthropic
- **다중 VectorDB**: ChromaDB (개발), Milvus (운영)
- **인증/권한**: JWT + RBAC (4단계 역할)
- **Agent Builder**: 시각적 워크플로우 편집기
- **MCP Tools**: 8개 내장 도구 (검색, 번역, 분석 등)

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| Backend | Python 3.11+, FastAPI, asyncio |
| Frontend | React 18, TypeScript, MUI, React Flow |
| LLM | vLLM, Ollama, OpenAI, Anthropic |
| Embedding | BAAI/bge-m3 (1024 dim) |
| Vector DB | ChromaDB, Milvus |
| Database | SQLite (개발), PostgreSQL (운영) |
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
# 샘플 문서 인제스트
python -c "
import asyncio
from pathlib import Path
from pipeline.ingest import IngestPipeline

async def main():
    pipeline = IngestPipeline()
    for f in Path('data/sample').glob('*.txt'):
        result = await pipeline.ingest(f)
        print(f'{f.name}: {result.chunk_count} chunks')

asyncio.run(main())
"
```

### 3. 서버 실행

```bash
# 백엔드 (포트 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 프론트엔드 (포트 5173)
cd ../frontend
npm install
npm run dev
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

## 프로젝트 구조

```
flux-rag/
├── backend/
│   ├── api/              # FastAPI 라우트
│   ├── agent/            # Agent, MCP, Builder
│   ├── auth/             # JWT, RBAC, LDAP
│   ├── config/           # Pydantic 설정
│   ├── core/
│   │   ├── llm/          # LLM 프로바이더
│   │   ├── embeddings/   # 임베딩 모델
│   │   ├── vectorstore/  # ChromaDB, Milvus
│   │   ├── cache/        # Redis, 메모리 캐시
│   │   └── performance/  # 배치 처리, 프로파일링
│   ├── pipeline/         # 문서 처리 파이프라인
│   ├── rag/              # RAG 체인, 검색, 품질
│   ├── monitoring/       # Prometheus 메트릭
│   └── tests/            # pytest 테스트
├── frontend/
│   ├── src/
│   │   ├── pages/        # 페이지 컴포넌트
│   │   ├── components/   # 공통 컴포넌트
│   │   ├── api/          # API 클라이언트
│   │   └── contexts/     # Auth Context
│   └── public/
├── k8s/                  # Kubernetes 매니페스트
├── docs/                 # 문서
└── docker-compose.yml
```

## API 엔드포인트

### Chat
- `POST /api/chat` - RAG 질의응답 (SSE 스트리밍)
- `GET /api/sessions` - 대화 세션 목록

### Documents
- `POST /api/documents/upload` - 문서 업로드
- `GET /api/documents` - 문서 목록
- `DELETE /api/documents/{id}` - 문서 삭제

### Admin
- `GET /api/admin/settings` - 설정 조회
- `PUT /api/admin/settings` - 설정 변경
- `GET /api/admin/stats` - 시스템 통계

### Auth
- `POST /api/auth/login` - 로그인
- `POST /api/auth/refresh` - 토큰 갱신
- `GET /api/auth/me` - 현재 사용자

## 환경 변수

```bash
# LLM
DEFAULT_LLM_PROVIDER=ollama  # vllm, ollama, openai, anthropic
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Auth
AUTH_ENABLED=false  # true for production
JWT_SECRET_KEY=your-secret-key

# Optional
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=false
PROMETHEUS_ENABLED=false
```

## Docker 배포

```bash
# 전체 스택 실행
docker-compose up -d

# 또는 개별 서비스
docker-compose up -d backend frontend
```

## 테스트

```bash
cd backend

# 단위 테스트
pytest tests/ -v

# 특정 테스트
pytest tests/test_rag.py -v

# 커버리지
pytest tests/ --cov=. --cov-report=html
```

## 라이선스

MIT License

## 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request
