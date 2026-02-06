# flux-rag 배포 가이드

이 문서는 flux-rag 프로젝트를 Docker 및 Kubernetes 환경에 배포하는 방법을 설명합니다.

## 목차

1. [Docker Compose로 로컬 배포](#docker-compose로-로컬-배포)
2. [Kubernetes로 프로덕션 배포](#kubernetes로-프로덕션-배포)
3. [환경 변수 설정](#환경-변수-설정)
4. [트러블슈팅](#트러블슈팅)

---

## Docker Compose로 로컬 배포

로컬 개발 및 테스트 환경에서 빠르게 실행할 수 있습니다.

### 사전 요구사항

- Docker 20.10 이상
- Docker Compose v2 이상

### 배포 단계

1. **환경 변수 설정**

   ```bash
   # .env.example을 .env로 복사
   cp .env.example .env

   # .env 파일 편집하여 API 키 입력
   nano .env
   ```

2. **Docker 이미지 빌드 및 실행**

   ```bash
   # 백그라운드에서 실행
   docker-compose up -d

   # 로그 확인
   docker-compose logs -f

   # 특정 서비스 로그만 확인
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

3. **접속**

   - 프론트엔드: http://localhost:3000
   - 백엔드 API: http://localhost:8000
   - API 문서: http://localhost:8000/docs

4. **종료**

   ```bash
   # 컨테이너 중지
   docker-compose stop

   # 컨테이너 삭제
   docker-compose down

   # 볼륨까지 삭제 (데이터 초기화)
   docker-compose down -v
   ```

### 개발 모드

소스 코드 변경을 실시간 반영하려면:

```yaml
# docker-compose.override.yml 생성
services:
  backend:
    volumes:
      - ./backend:/app
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    volumes:
      - ./frontend/src:/app/src
```

```bash
docker-compose up -d
```

---

## Kubernetes로 프로덕션 배포

프로덕션 환경에서 고가용성 및 확장성을 위한 배포 방법입니다.

### 사전 요구사항

- Kubernetes 클러스터 (v1.20 이상)
- kubectl CLI 도구
- Docker 이미지 레지스트리 (Docker Hub, ghcr.io, ECR 등)

### 배포 단계

상세한 내용은 [k8s/README.md](k8s/README.md)를 참조하세요.

1. **Docker 이미지 빌드 및 푸시**

   ```bash
   # 백엔드
   cd backend
   docker build -t your-registry/flux-rag-backend:v1.0.0 .
   docker push your-registry/flux-rag-backend:v1.0.0

   # 프론트엔드
   cd frontend
   docker build -t your-registry/flux-rag-frontend:v1.0.0 .
   docker push your-registry/flux-rag-frontend:v1.0.0
   ```

2. **Kubernetes 설정 파일 수정**

   ```bash
   # Secret 생성
   kubectl create secret generic flux-rag-secret \
     --from-literal=VLLM_API_KEY='your-key' \
     --from-literal=OPENAI_API_KEY='your-key' \
     --from-literal=SECRET_KEY='your-secret' \
     --namespace=flux-rag

   # Deployment 파일에서 이미지 경로 수정
   # backend-deployment.yaml, frontend-deployment.yaml
   ```

3. **배포 실행**

   ```bash
   kubectl apply -f k8s/
   ```

4. **배포 확인**

   ```bash
   kubectl get pods -n flux-rag
   kubectl get svc -n flux-rag
   kubectl get ingress -n flux-rag
   ```

---

## 환경 변수 설정

### 필수 환경 변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `SECRET_KEY` | JWT 토큰 암호화 키 | `your-secret-key` |
| `DEFAULT_LLM_PROVIDER` | 기본 LLM 제공자 | `vllm`, `ollama`, `openai`, `anthropic` |

### LLM Provider별 설정

#### vLLM (RunPod 등 원격 서버)

```env
VLLM_BASE_URL=http://your-vllm-server:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_API_KEY=your-vllm-api-key
DEFAULT_LLM_PROVIDER=vllm
```

#### Ollama (로컬)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
DEFAULT_LLM_PROVIDER=ollama
```

#### OpenAI

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
DEFAULT_LLM_PROVIDER=openai
```

#### Anthropic

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
DEFAULT_LLM_PROVIDER=anthropic
```

### Embeddings 설정

```env
EMBEDDING_MODEL=BAAI/bge-m3
```

지원 모델:
- `BAAI/bge-m3` (다국어, 권장)
- `BAAI/bge-base-en-v1.5` (영어)
- `sentence-transformers/all-MiniLM-L6-v2` (경량)

### 데이터베이스 설정

```env
CHROMA_PERSIST_DIR=./data/chroma_db
DATABASE_URL=sqlite+aiosqlite:///./data/sqlite.db
```

---

## 트러블슈팅

### Docker Compose 관련

#### 1. 포트 충돌

```bash
# 에러: Bind for 0.0.0.0:8000 failed: port is already allocated
# 해결: 사용 중인 포트 변경
docker-compose down
# docker-compose.yml에서 포트 변경 (예: 8001:8000)
docker-compose up -d
```

#### 2. 볼륨 권한 문제

```bash
# 에러: Permission denied
# 해결: 볼륨 디렉토리 권한 수정
sudo chown -R 1000:1000 ./backend/data
```

#### 3. 메모리 부족

```bash
# 에러: Killed (OOM)
# 해결: Docker Desktop 메모리 제한 증가 (최소 4GB 권장)
# 또는 docker-compose.yml에서 제한 설정
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Kubernetes 관련

#### 1. ImagePullBackOff

```bash
# 문제: 이미지를 가져올 수 없음
kubectl describe pod <pod-name> -n flux-rag

# 해결 1: 이미지 경로 확인
# 해결 2: 프라이빗 레지스트리 인증 설정
kubectl create secret docker-registry regcred \
  --docker-server=your-registry \
  --docker-username=your-user \
  --docker-password=your-password \
  --namespace=flux-rag

# Deployment에 imagePullSecrets 추가
spec:
  template:
    spec:
      imagePullSecrets:
      - name: regcred
```

#### 2. CrashLoopBackOff

```bash
# 로그 확인
kubectl logs <pod-name> -n flux-rag --previous

# 일반적인 원인:
# - 환경 변수 누락 (SECRET_KEY, API 키 등)
# - 데이터베이스 연결 실패
# - 의존성 패키지 문제
```

#### 3. PVC Pending

```bash
# StorageClass 확인
kubectl get storageclass

# 해결: pvc.yaml에 storageClassName 추가
spec:
  storageClassName: standard  # 클러스터의 기본 StorageClass
```

### 백엔드 관련

#### 1. 모델 다운로드 실패

```bash
# sentence-transformers 모델 수동 다운로드
docker exec -it flux-rag-backend bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

#### 2. ChromaDB 초기화 오류

```bash
# 데이터 디렉토리 삭제 후 재시작
rm -rf backend/data/chroma_db
docker-compose restart backend
```

### 프론트엔드 관련

#### 1. API 연결 실패

```bash
# 백엔드 URL 확인
# nginx.conf의 proxy_pass 확인
# CORS 설정 확인 (backend/api/main.py)
```

#### 2. 빌드 실패

```bash
# Node.js 버전 확인 (18 이상 필요)
node --version

# 의존성 재설치
rm -rf node_modules package-lock.json
npm install
```

---

## 프로덕션 체크리스트

배포 전 확인 사항:

- [ ] `.env` 파일의 모든 API 키 설정
- [ ] `SECRET_KEY`를 강력한 랜덤 문자열로 변경
- [ ] Docker 이미지에 버전 태그 사용 (`:latest` 지양)
- [ ] 헬스체크 엔드포인트 동작 확인
- [ ] 리소스 제한 설정 (CPU, 메모리)
- [ ] 데이터 볼륨 백업 계획 수립
- [ ] 모니터링 및 로깅 설정 (Prometheus, Grafana 등)
- [ ] TLS/HTTPS 인증서 설정
- [ ] 보안 스캔 (Trivy, Snyk 등)

---

## 참고 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Kubernetes 공식 문서](https://kubernetes.io/docs/)
- [FastAPI 배포 가이드](https://fastapi.tiangolo.com/deployment/)
- [Vite 프로덕션 빌드](https://vitejs.dev/guide/build.html)
