#!/usr/bin/env bash
# flux-rag 오프라인 설치 패키지 빌드
#
# 용도: 폐쇄망(인터넷 미연결) 환경에 배포할 패키지 생성
# 실행: bash scripts/offline-package/build-package.sh
#
# 출력: dist/flux-rag-offline-{date}.tar.gz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE=$(date +%Y%m%d)
OUTPUT_DIR="$PROJECT_ROOT/dist/flux-rag-offline-$DATE"

echo "=== flux-rag 오프라인 패키지 빌드 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "출력 디렉토리: $OUTPUT_DIR"

# Clean & create
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{docker-images,python-wheels,frontend-dist,config,scripts,models}

# 1. Docker 이미지 저장
echo ""
echo "[1/6] Docker 이미지 저장..."
DOCKER_IMAGES=(
    "python:3.11-slim"
    "node:20-alpine"
    "redis:7-alpine"
    "chromadb/chroma:latest"
    "prom/prometheus:latest"
    "grafana/grafana:latest"
)

for img in "${DOCKER_IMAGES[@]}"; do
    echo "  Pulling & saving: $img"
    docker pull "$img" 2>/dev/null || echo "  WARN: $img not pulled (may already exist)"
    SAFE_NAME=$(echo "$img" | tr '/:' '_')
    docker save "$img" | gzip > "$OUTPUT_DIR/docker-images/${SAFE_NAME}.tar.gz"
done
echo "  Docker 이미지 저장 완료: $(ls "$OUTPUT_DIR/docker-images/" | wc -l) 이미지"

# 2. Python 패키지 (wheels)
echo ""
echo "[2/6] Python 패키지 빌드..."
cd "$PROJECT_ROOT/backend"

# Poetry export로 requirements.txt 생성
if command -v poetry &> /dev/null; then
    echo "  Poetry로 의존성 추출 중..."
    poetry export -f requirements.txt --without-hashes --output /tmp/flux-rag-requirements.txt 2>/dev/null || {
        echo "  WARN: Poetry export 실패, pyproject.toml에서 직접 추출..."
        # Poetry 없으면 pip freeze 사용
        python3 -m pip freeze > /tmp/flux-rag-requirements.txt
    }
else
    echo "  Poetry 미설치, pip freeze 사용..."
    python3 -m pip freeze > /tmp/flux-rag-requirements.txt
fi

# Python 플랫폼별 wheel 다운로드
echo "  Python 패키지 다운로드 중..."
pip download \
    -r /tmp/flux-rag-requirements.txt \
    -d "$OUTPUT_DIR/python-wheels" \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --only-binary=:all: 2>/dev/null || {
    echo "  WARN: 플랫폼 필터링 실패, 로컬 빌드 시도..."
    pip wheel --wheel-dir="$OUTPUT_DIR/python-wheels" -r /tmp/flux-rag-requirements.txt 2>/dev/null || \
        echo "  WARN: wheel 빌드 실패"
}

WHEEL_COUNT=$(find "$OUTPUT_DIR/python-wheels" -name "*.whl" 2>/dev/null | wc -l)
echo "  Python 패키지 수: $WHEEL_COUNT"

# 3. 임베딩 모델 다운로드 (선택)
echo ""
echo "[3/6] 임베딩 모델 다운로드..."
EMBEDDING_MODEL="BAAI/bge-m3"
if python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')" 2>/dev/null; then
    echo "  모델 다운로드 완료: $EMBEDDING_MODEL"
    # 모델 파일 복사
    MODEL_CACHE="$HOME/.cache/huggingface/hub"
    if [ -d "$MODEL_CACHE" ]; then
        mkdir -p "$OUTPUT_DIR/models/embeddings"
        find "$MODEL_CACHE" -name "*bge-m3*" -type d -exec cp -r {} "$OUTPUT_DIR/models/embeddings/" \; 2>/dev/null || true
    fi
else
    echo "  WARN: 임베딩 모델 다운로드 실패 (인터넷 연결 필요)"
fi

# 4. 프론트엔드 빌드
echo ""
echo "[4/6] 프론트엔드 빌드..."
cd "$PROJECT_ROOT/frontend"
if [ -f "package.json" ]; then
    echo "  npm 의존성 설치 중..."
    npm ci --ignore-scripts 2>/dev/null || npm install

    echo "  프론트엔드 빌드 중..."
    npm run build 2>/dev/null || {
        echo "  WARN: Frontend build 실패"
    }

    if [ -d "dist" ]; then
        cp -r dist/* "$OUTPUT_DIR/frontend-dist/"
        echo "  프론트엔드 빌드 완료"
    fi

    # npm 오프라인 캐시 생성
    echo "  npm 오프라인 캐시 생성 중..."
    npm pack --pack-destination "$OUTPUT_DIR/frontend-dist/" 2>/dev/null || true

    # node_modules도 포함 (옵션)
    if [ -d "node_modules" ]; then
        echo "  node_modules 아카이브 중..."
        tar czf "$OUTPUT_DIR/frontend-dist/node_modules.tar.gz" node_modules 2>/dev/null || true
    fi
else
    echo "  WARN: package.json 없음, 프론트엔드 스킵"
fi

# 5. 설정 파일 및 소스 복사
echo ""
echo "[5/6] 설정 파일 및 소스 복사..."
cd "$PROJECT_ROOT"

# 백엔드 소스 코드
echo "  백엔드 소스 복사 중..."
mkdir -p "$OUTPUT_DIR/backend"
rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='data/chroma_db' --exclude='data/memory.db' \
    "$PROJECT_ROOT/backend/" "$OUTPUT_DIR/backend/" 2>/dev/null || \
    cp -r "$PROJECT_ROOT/backend" "$OUTPUT_DIR/" 2>/dev/null || true

# 환경 설정
if [ -f "backend/.env.example" ]; then
    cp "backend/.env.example" "$OUTPUT_DIR/config/.env.example"
else
    cat > "$OUTPUT_DIR/config/.env.example" <<'EOF'
# flux-rag 환경설정 (production 값으로 수정 필요)

# LLM Provider
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# vLLM (운영 환경)
VLLM_BASE_URL=http://gpu-server:8000/v1
VLLM_MODEL=meta-llama/Llama-3-8B-Instruct

# Vector Database
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Authentication
AUTH_ENABLED=true
JWT_SECRET_KEY=CHANGE_THIS_SECRET_KEY
JWT_ALGORITHM=HS256

# OCR
OCR_PROVIDER=onprem
OCR_ONPREM_URL=http://localhost:8501

# Redis
REDIS_URL=redis://localhost:6379/0
EOF
fi

# Docker Compose 파일
cp -f "backend/docker-compose.monitoring.yml" "$OUTPUT_DIR/config/" 2>/dev/null || true
cp -f "backend/docker-compose.ocr.yml" "$OUTPUT_DIR/config/" 2>/dev/null || true

# 프롬프트 템플릿
if [ -d "backend/prompts" ]; then
    cp -r "backend/prompts" "$OUTPUT_DIR/config/prompts"
fi

# 벤치마크 데이터 (샘플)
if [ -d "backend/data/sample_dataset" ]; then
    echo "  벤치마크 데이터 복사 중..."
    cp -r "backend/data/sample_dataset" "$OUTPUT_DIR/backend/data/" 2>/dev/null || true
fi

# README 및 문서
cp -f "README.md" "$OUTPUT_DIR/" 2>/dev/null || true
cp -f "CLAUDE.md" "$OUTPUT_DIR/" 2>/dev/null || true

echo "  설정 파일 복사 완료"

# 6. 설치 스크립트 복사
echo ""
echo "[6/6] 설치 스크립트 복사..."
cp "$SCRIPT_DIR/install.sh" "$OUTPUT_DIR/scripts/"
chmod +x "$OUTPUT_DIR/scripts/install.sh"

if [ -f "$SCRIPT_DIR/README_INSTALL.md" ]; then
    cp "$SCRIPT_DIR/README_INSTALL.md" "$OUTPUT_DIR/"
else
    echo "  WARN: README_INSTALL.md 없음"
fi

# 패키징
echo ""
echo "=== 최종 패키지 생성 ==="
cd "$PROJECT_ROOT/dist"
echo "  압축 중..."
tar czf "flux-rag-offline-$DATE.tar.gz" "flux-rag-offline-$DATE/"
FINAL_SIZE=$(du -sh "flux-rag-offline-$DATE.tar.gz" | cut -f1)
FINAL_PATH="$PROJECT_ROOT/dist/flux-rag-offline-$DATE.tar.gz"

echo ""
echo "=========================================="
echo "패키지 생성 완료!"
echo "=========================================="
echo "파일: $FINAL_PATH"
echo "크기: $FINAL_SIZE"
echo ""
echo "패키지 내용:"
echo "  - Docker 이미지: $(ls "$OUTPUT_DIR/docker-images/" 2>/dev/null | wc -l)개"
echo "  - Python wheels: $WHEEL_COUNT개"
echo "  - 프론트엔드: $([ -d "$OUTPUT_DIR/frontend-dist" ] && echo "빌드됨" || echo "없음")"
echo "  - 백엔드 소스: $([ -d "$OUTPUT_DIR/backend" ] && echo "포함됨" || echo "없음")"
echo ""
echo "다음 단계:"
echo "  1. 파일을 폐쇄망 서버로 전송"
echo "  2. 압축 해제 후 scripts/install.sh 실행"
echo "=========================================="

# 정리
rm -rf "$OUTPUT_DIR"
rm -f /tmp/flux-rag-requirements.txt
echo "Done!"
