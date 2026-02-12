# flux-rag 오프라인 설치 가이드

## 개요

이 패키지는 폐쇄망(Air-Gapped) 환경에서 flux-rag를 설치하기 위한 완전한 오프라인 설치 패키지입니다.

**패키지 내용:**
- Docker 이미지 (Redis, ChromaDB, Prometheus, Grafana)
- Python 패키지 (wheels)
- 프론트엔드 빌드 파일
- 백엔드 소스 코드
- 설정 파일 및 템플릿
- 임베딩 모델 (선택)

---

## 사전 요구사항

### 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| OS | CentOS 7+ / Ubuntu 20.04+ | Rocky Linux 9 / Ubuntu 22.04 |
| CPU | 8 코어 | 16+ 코어 |
| RAM | 16GB | 32GB+ |
| GPU | - | NVIDIA A100 (vLLM 서빙용) |
| 디스크 | 100GB | 500GB+ SSD |

### 소프트웨어 요구사항

**필수:**
- Docker 24.0+
- Docker Compose 2.0+
- Python 3.11+

**선택:**
- Node.js 20+ (프론트엔드 개발 시)
- nginx (프론트엔드 서빙용)
- NVIDIA Docker Runtime (GPU 사용 시)

### 설치 확인

```bash
# Docker 확인
docker --version
docker compose version

# Python 확인
python3.11 --version

# GPU 확인 (선택)
nvidia-smi
```

---

## 설치 순서

### 1. 패키지 전송

폐쇄망 서버로 패키지를 전송합니다.

**방법 1: USB 메모리**
```bash
# 외부에서 USB로 복사
cp flux-rag-offline-YYYYMMDD.tar.gz /media/usb/

# 폐쇄망 서버에서 복사
cp /media/usb/flux-rag-offline-YYYYMMDD.tar.gz /tmp/
```

**방법 2: 내부 파일서버**
```bash
# 내부 파일서버에서 다운로드
wget http://internal-fileserver/flux-rag-offline-YYYYMMDD.tar.gz
# 또는
scp user@fileserver:/path/flux-rag-offline-YYYYMMDD.tar.gz /tmp/
```

### 2. 패키지 압축 해제

```bash
cd /tmp
tar xzf flux-rag-offline-YYYYMMDD.tar.gz
cd flux-rag-offline-YYYYMMDD
```

**압축 해제 확인:**
```bash
ls -lh
# 출력:
# docker-images/       # Docker 이미지
# python-wheels/       # Python 패키지
# frontend-dist/       # 프론트엔드
# backend/             # 백엔드 소스
# config/              # 설정 파일
# scripts/             # 설치 스크립트
# models/              # 임베딩 모델
```

### 3. 설치 실행

**기본 설치 (권장 경로: /opt/flux-rag):**
```bash
sudo bash scripts/install.sh
```

**커스텀 경로 설치:**
```bash
INSTALL_DIR=/home/kogas/flux-rag sudo bash scripts/install.sh
```

**설치 과정:**
1. Docker 이미지 로드
2. Python 가상환경 생성 및 패키지 설치
3. 백엔드 소스 복사
4. 프론트엔드 배포
5. 설정 파일 배치

**예상 시간:** 10-20분 (하드웨어 성능에 따라)

### 4. 환경 설정

```bash
vi /opt/flux-rag/backend/.env
```

**주요 설정 항목:**

```bash
# LLM Provider 선택
DEFAULT_LLM_PROVIDER=vllm        # 운영 환경 (고성능)
# DEFAULT_LLM_PROVIDER=ollama    # 개발 환경 (로컬)

# vLLM 설정 (운영 환경)
VLLM_BASE_URL=http://gpu-server:8000/v1
VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
VLLM_API_KEY=your-api-key

# Ollama 설정 (개발 환경)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# 인증 활성화 (운영 환경 필수)
AUTH_ENABLED=true
JWT_SECRET_KEY=CHANGE_THIS_TO_RANDOM_SECRET_KEY  # 반드시 변경!
JWT_ALGORITHM=HS256

# OCR 설정 (선택)
OCR_PROVIDER=onprem
OCR_ONPREM_URL=http://localhost:8501

# Redis 설정
REDIS_URL=redis://localhost:6379/0

# Vector Store
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base
```

**JWT Secret Key 생성:**
```bash
openssl rand -hex 32
```

### 5. Docker 서비스 시작

**모니터링 스택 (Redis, Prometheus, Grafana):**
```bash
cd /opt/flux-rag
docker compose -f docker-compose.monitoring.yml up -d
```

**서비스 확인:**
```bash
docker compose -f docker-compose.monitoring.yml ps

# 출력:
# NAME                  STATUS
# flux-rag-redis        Up (healthy)
# flux-rag-prometheus   Up (healthy)
# flux-rag-grafana      Up (healthy)
```

**OCR 온프레미스 (선택):**
```bash
# GPU 환경
docker compose -f docker-compose.ocr.yml up -d

# CPU 환경
docker compose -f docker-compose.ocr.yml --profile cpu-only up -d
```

### 6. 백엔드 실행

**방법 1: 직접 실행 (개발/테스트)**
```bash
cd /opt/flux-rag/backend
source ../.venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**방법 2: systemd 서비스 (운영 환경 권장)**
```bash
# 서비스 파일 복사
sudo cp /tmp/flux-rag-backend.service /etc/systemd/system/

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable flux-rag-backend.service
sudo systemctl start flux-rag-backend.service

# 상태 확인
sudo systemctl status flux-rag-backend.service

# 로그 확인
sudo journalctl -u flux-rag-backend.service -f
```

### 7. 프론트엔드 배포 (nginx 권장)

**nginx 설정 파일 생성:**
```bash
sudo vi /etc/nginx/conf.d/flux-rag.conf
```

**설정 내용:**
```nginx
server {
    listen 80;
    server_name flux-rag.kogas.local;

    # 프론트엔드 (React SPA)
    location / {
        root /opt/flux-rag/frontend;
        try_files $uri $uri/ /index.html;

        # 캐시 설정
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # 백엔드 API 프록시
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 타임아웃 설정 (RAG 응답 시간 고려)
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }

    # 헬스체크
    location /health {
        proxy_pass http://localhost:8000;
        access_log off;
    }
}
```

**nginx 재시작:**
```bash
sudo nginx -t  # 설정 검증
sudo systemctl restart nginx
```

### 8. 검증

**헬스체크:**
```bash
curl http://localhost:8000/health
# 출력: {"status":"ok","version":"0.1.0"}
```

**문서 업로드 테스트:**
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

**RAG 질의 테스트:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "한국가스공사의 목적은 무엇인가?",
    "session_id": "test-session"
  }'
```

**프론트엔드 접속:**
```bash
# 브라우저에서
http://flux-rag.kogas.local

# 또는
http://localhost
```

---

## 초기 데이터 설정

### 1. 벤치마크 데이터 인제스트

```bash
cd /opt/flux-rag/backend
source ../.venv/bin/activate

# 한국가스공사법 데이터
python -c "
import asyncio
from pathlib import Path
from pipeline.ingest import IngestPipeline

async def main():
    pipeline = IngestPipeline()
    law_dir = Path('data/sample_dataset/한국가스공사법')

    for pdf_file in law_dir.glob('*.pdf'):
        print(f'Processing: {pdf_file.name}')
        result = await pipeline.ingest(pdf_file)
        print(f'  Chunks: {result.chunk_count}')

asyncio.run(main())
"
```

### 2. 사용자 계정 생성

```bash
# 관리자 계정 생성
python -c "
from core.auth import create_user
import asyncio

async def main():
    await create_user('admin', 'secure_password', role='admin')
    print('Admin user created')

asyncio.run(main())
"
```

---

## 모니터링

### Grafana 대시보드

**접속:**
- URL: http://localhost:3001
- 계정: admin / admin

**초기 설정:**
1. 로그인 후 비밀번호 변경
2. Data Source 추가: Prometheus (http://prometheus:9090)
3. 대시보드 임포트 (ID: 7587 - FastAPI 모니터링)

### Prometheus 메트릭

**접속:**
- URL: http://localhost:9090

**주요 메트릭:**
- `http_requests_total` - 요청 수
- `http_request_duration_seconds` - 응답 시간
- `rag_query_latency_seconds` - RAG 쿼리 지연시간
- `vectorstore_size` - 벡터 저장소 크기

---

## 문제 해결

### Docker 이미지 로드 실패

**증상:** "Error loading image from..."

**해결:**
```bash
# 수동 로드
cd /tmp/flux-rag-offline-YYYYMMDD/docker-images
for img in *.tar.gz; do
    echo "Loading $img..."
    docker load < "$img"
done

# 이미지 확인
docker images
```

### pip 설치 오류

**증상:** "Could not find a version that satisfies..."

**해결:**
```bash
source /opt/flux-rag/.venv/bin/activate

# 특정 패키지 수동 설치
pip install --no-index \
    --find-links=/tmp/flux-rag-offline-YYYYMMDD/python-wheels \
    package_name
```

### GPU 인식 안됨

**증상:** "RuntimeError: CUDA out of memory" 또는 GPU 미사용

**해결:**
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# NVIDIA Container Toolkit 설치 (필요 시)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### ChromaDB 연결 실패

**증상:** "Could not connect to ChromaDB"

**해결:**
```bash
# ChromaDB 컨테이너 상태 확인
docker ps | grep chroma

# ChromaDB 로그 확인
docker logs flux-rag-chroma

# ChromaDB 재시작
docker compose -f docker-compose.monitoring.yml restart chromadb

# 데이터 초기화 (주의: 모든 데이터 삭제)
sudo rm -rf /opt/flux-rag/backend/data/chroma_db
docker compose -f docker-compose.monitoring.yml restart chromadb
```

### 메모리 부족

**증상:** "MemoryError" 또는 OOM Killer

**해결:**
```bash
# Redis 메모리 제한 확인
docker inspect flux-rag-redis | grep -A 5 Memory

# 워커 수 감소
# uvicorn --workers 2  (기본 4에서 감소)

# 임베딩 모델 경량화
# .env 파일에서:
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  (384 dim)
```

### OCR 서비스 실패

**증상:** "OCR service unavailable"

**해결:**
```bash
# OCR 컨테이너 확인
docker ps | grep ocr

# 로그 확인
docker logs flux-rag-ocr

# GPU 메모리 확인
nvidia-smi

# CPU 모드로 전환 (GPU 부족 시)
docker compose -f docker-compose.ocr.yml down
docker compose -f docker-compose.ocr.yml --profile cpu-only up -d
```

### 포트 충돌

**증상:** "Address already in use"

**해결:**
```bash
# 포트 사용 확인
sudo netstat -tulpn | grep :8000

# 프로세스 종료
sudo kill -9 <PID>

# 또는 다른 포트 사용
uvicorn api.main:app --port 8001
```

---

## 성능 튜닝

### 1. uvicorn 워커 수 조정

```bash
# CPU 코어 수 확인
nproc

# 권장: CPU 코어 수 - 1
uvicorn api.main:app --workers $(expr $(nproc) - 1)
```

### 2. Redis 메모리 최적화

```bash
# docker-compose.monitoring.yml 수정
command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### 3. ChromaDB 인덱스 튜닝

```python
# config/settings.py 수정
CHROMA_HNSW_SPACE = "cosine"
CHROMA_HNSW_EF_CONSTRUCTION = 200
CHROMA_HNSW_EF_SEARCH = 50
CHROMA_HNSW_M = 16
```

### 4. 벡터 검색 성능

```python
# .env 파일
RAG_TOP_K=10              # 초기 검색 수 (20→10)
RAG_RERANK_TOP_K=3        # 재순위 후 (5→3)
RAG_SIMILARITY_THRESHOLD=0.6  # 유사도 임계값
```

---

## 보안 권고사항

### 1. 환경 변수 보안

```bash
# .env 파일 권한 설정
chmod 600 /opt/flux-rag/backend/.env

# JWT Secret 강화
openssl rand -hex 64  # 128자 생성
```

### 2. 방화벽 설정

```bash
# 필수 포트만 개방
sudo firewall-cmd --permanent --add-port=80/tcp    # nginx
sudo firewall-cmd --permanent --add-port=443/tcp   # HTTPS
sudo firewall-cmd --reload

# 내부 서비스는 외부 접근 차단
# 8000 (FastAPI), 6379 (Redis), 9090 (Prometheus) 등
```

### 3. HTTPS 활성화

```bash
# Let's Encrypt 인증서 (인터넷 연결 필요)
# 또는 자체 서명 인증서
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/flux-rag.key \
    -out /etc/ssl/certs/flux-rag.crt
```

### 4. 로그 관리

```bash
# 로그 디렉토리 생성
mkdir -p /var/log/flux-rag

# logrotate 설정
sudo vi /etc/logrotate.d/flux-rag
```

---

## 업그레이드

새 버전 패키지로 업그레이드:

```bash
# 1. 백업
sudo tar czf /backup/flux-rag-backup-$(date +%Y%m%d).tar.gz /opt/flux-rag

# 2. 서비스 중지
sudo systemctl stop flux-rag-backend
docker compose -f /opt/flux-rag/docker-compose.monitoring.yml down

# 3. 새 패키지 설치
cd /tmp
tar xzf flux-rag-offline-YYYYMMDD.tar.gz
cd flux-rag-offline-YYYYMMDD
INSTALL_DIR=/opt/flux-rag sudo bash scripts/install.sh

# 4. 설정 파일 복원 (.env 등)
# 5. 서비스 재시작
```

---

## 연락처 및 지원

**문제 보고:**
- 내부 이슈 트래커: http://jira.kogas.local/flux-rag
- 이메일: flux-rag-support@kogas.co.kr

**문서:**
- 사용자 매뉴얼: /opt/flux-rag/CLAUDE.md
- API 문서: http://localhost:8000/docs

---

## 부록

### A. 디렉토리 구조

```
/opt/flux-rag/
├── .venv/                          # Python 가상환경
├── backend/
│   ├── api/                        # FastAPI 라우트
│   ├── agent/                      # Agent, MCP
│   ├── core/                       # 핵심 컴포넌트
│   ├── pipeline/                   # 문서 처리
│   ├── rag/                        # RAG 체인
│   ├── data/
│   │   ├── chroma_db/              # 벡터 저장소
│   │   ├── uploads/                # 업로드 파일
│   │   └── sample_dataset/         # 샘플 데이터
│   ├── prompts/                    # 프롬프트 템플릿
│   └── .env                        # 환경 설정
├── frontend/                       # 프론트엔드 빌드
│   ├── index.html
│   ├── assets/
│   └── ...
├── docker-compose.monitoring.yml   # 모니터링 스택
└── docker-compose.ocr.yml          # OCR 서비스
```

### B. 환경 변수 전체 목록

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `DEFAULT_LLM_PROVIDER` | ollama | LLM 프로바이더 |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama URL |
| `VLLM_BASE_URL` | - | vLLM URL |
| `AUTH_ENABLED` | false | 인증 활성화 |
| `JWT_SECRET_KEY` | - | JWT 시크릿 |
| `REDIS_URL` | redis://localhost:6379/0 | Redis URL |
| `CHROMA_PERSIST_DIR` | ./data/chroma_db | ChromaDB 경로 |
| `OCR_PROVIDER` | none | OCR 프로바이더 |
| `RAG_TOP_K` | 20 | 초기 검색 수 |
| `RAG_RERANK_TOP_K` | 5 | 재순위 수 |

### C. 테스트 계정

| 역할 | 사용자명 | 비밀번호 |
|------|---------|---------|
| Admin | admin | admin123 |
| Manager | manager | manager123 |
| User | user | user123 |

**주의:** 운영 환경에서는 반드시 비밀번호를 변경하세요!

---

**버전:** 0.1.0
**최종 수정:** 2026-02-09
**작성자:** flux-rag 개발팀
