# Flux RAG - 프로젝트 가이드

## 프로젝트 개요
한국가스기술공사 생성형 AI 플랫폼 - RAG(Retrieval-Augmented Generation) 프레임워크

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
python tests/benchmark_kogas_law.py           # 벤치마크 (50문항)
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
```

## 프로젝트 구조

```
flux-rag/
├── backend/
│   ├── api/              # FastAPI 라우트
│   ├── agent/            # Agent, MCP, Router
│   ├── config/           # Pydantic 설정
│   ├── core/
│   │   ├── llm/          # LLM 프로바이더 (Ollama, OpenAI, Anthropic, vLLM)
│   │   ├── embeddings/   # 임베딩 (bge-m3)
│   │   └── vectorstore/  # ChromaDB, Milvus
│   ├── pipeline/         # 문서 처리 파이프라인
│   ├── rag/              # RAG 체인, 검색, 품질
│   ├── data/
│   │   └── sample_dataset/한국가스공사법/  # 벤치마크 데이터
│   └── tests/
├── frontend/             # React + TypeScript + MUI
└── k8s/                  # Kubernetes 배포
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

## 벤치마크 결과

한국가스공사법 50문항 테스트:
- 성공률: 100% (50/50)
- 평균 신뢰도: 0.999
- 카테고리: factual(20), inference(15), multi_hop(10), negative(5)

## 테스트 계정

| 역할 | 사용자명 | 비밀번호 |
|------|---------|---------|
| Admin | admin | admin123 |
| Manager | manager | manager123 |
| User | user | user123 |

## API 엔드포인트

- `POST /api/chat` - RAG 질의응답
- `POST /api/documents/upload` - 문서 업로드
- `GET /api/documents` - 문서 목록
- `GET /health` - 헬스 체크

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
