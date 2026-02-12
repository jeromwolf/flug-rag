#!/usr/bin/env bash
# flux-rag 오프라인 설치 스크립트
#
# 사전 요구사항:
#   - Docker + Docker Compose 설치됨
#   - Python 3.11+ 설치됨
#   - Node.js 20+ 설치됨 (프론트엔드 서빙 시)
#
# 실행: bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/opt/flux-rag}"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=== flux-rag 오프라인 설치 ==="
echo "패키지 디렉토리: $PACKAGE_DIR"
echo "설치 디렉토리: $INSTALL_DIR"
echo ""

# 사전 요구사항 체크
log_info "사전 요구사항 체크..."
MISSING_DEPS=0

if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다"
    MISSING_DEPS=1
else
    log_info "Docker: $(docker --version)"
fi

if ! command -v python3 &> /dev/null && ! command -v python3.11 &> /dev/null; then
    log_error "Python 3.11+가 설치되지 않았습니다"
    MISSING_DEPS=1
else
    PYTHON_CMD=$(command -v python3.11 2>/dev/null || command -v python3)
    PYTHON_VERSION=$($PYTHON_CMD --version)
    log_info "Python: $PYTHON_VERSION"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    log_error "필수 의존성이 설치되지 않았습니다. 설치를 중단합니다."
    exit 1
fi

# 설치 디렉토리 생성
log_info "설치 디렉토리 생성..."
mkdir -p "$INSTALL_DIR"

# 1. Docker 이미지 로드
echo ""
log_info "[1/5] Docker 이미지 로드..."
if [ -d "$PACKAGE_DIR/docker-images" ]; then
    IMAGE_COUNT=0
    for img_file in "$PACKAGE_DIR/docker-images/"*.tar.gz; do
        if [ -f "$img_file" ]; then
            log_info "  Loading: $(basename "$img_file")"
            docker load < "$img_file" || log_warn "  이미지 로드 실패: $img_file"
            IMAGE_COUNT=$((IMAGE_COUNT + 1))
        fi
    done
    log_info "Docker 이미지 로드 완료: $IMAGE_COUNT개"
else
    log_warn "Docker 이미지 디렉토리 없음, 스킵"
fi

# 2. Python 환경 설정
echo ""
log_info "[2/5] Python 환경 설정..."
PYTHON_CMD=$(command -v python3.11 2>/dev/null || command -v python3)

log_info "  가상환경 생성 중: $INSTALL_DIR/.venv"
$PYTHON_CMD -m venv "$INSTALL_DIR/.venv"
source "$INSTALL_DIR/.venv/bin/activate"

log_info "  pip 업그레이드 중..."
pip install --upgrade pip setuptools wheel || log_warn "pip 업그레이드 실패"

if [ -d "$PACKAGE_DIR/python-wheels" ]; then
    WHEEL_COUNT=$(find "$PACKAGE_DIR/python-wheels" -name "*.whl" 2>/dev/null | wc -l)
    if [ "$WHEEL_COUNT" -gt 0 ]; then
        log_info "  Python 패키지 설치 중 ($WHEEL_COUNT개)..."
        pip install --no-index --find-links="$PACKAGE_DIR/python-wheels" "$PACKAGE_DIR/python-wheels/"*.whl || \
            log_warn "일부 패키지 설치 실패"
        log_info "Python 패키지 설치 완료"
    else
        log_warn "Python wheels 없음, 스킵"
    fi
else
    log_warn "Python wheels 디렉토리 없음, 스킵"
fi

# 3. 백엔드 소스 복사
echo ""
log_info "[3/5] 백엔드 소스 복사..."
if [ -d "$PACKAGE_DIR/backend" ]; then
    log_info "  백엔드 소스 복사 중..."
    mkdir -p "$INSTALL_DIR/backend"
    rsync -a "$PACKAGE_DIR/backend/" "$INSTALL_DIR/backend/" 2>/dev/null || \
        cp -r "$PACKAGE_DIR/backend/"* "$INSTALL_DIR/backend/" 2>/dev/null || \
        log_warn "백엔드 소스 복사 실패"

    # 데이터 디렉토리 생성
    mkdir -p "$INSTALL_DIR/backend/data"
    mkdir -p "$INSTALL_DIR/backend/data/chroma_db"
    mkdir -p "$INSTALL_DIR/backend/data/uploads"

    log_info "백엔드 소스 복사 완료"
else
    log_warn "백엔드 소스 디렉토리 없음, 스킵"
fi

# 4. 프론트엔드 배포
echo ""
log_info "[4/5] 프론트엔드 배포..."
if [ -d "$PACKAGE_DIR/frontend-dist" ]; then
    log_info "  프론트엔드 배포 중..."
    mkdir -p "$INSTALL_DIR/frontend"

    # 빌드된 파일 복사
    if [ "$(ls -A "$PACKAGE_DIR/frontend-dist" 2>/dev/null)" ]; then
        cp -r "$PACKAGE_DIR/frontend-dist/"* "$INSTALL_DIR/frontend/" 2>/dev/null || \
            log_warn "프론트엔드 파일 복사 실패"
        log_info "프론트엔드 배포 완료 ($INSTALL_DIR/frontend)"
    else
        log_warn "프론트엔드 빌드 파일 없음"
    fi

    # node_modules 압축 파일 있으면 풀기
    if [ -f "$PACKAGE_DIR/frontend-dist/node_modules.tar.gz" ]; then
        log_info "  node_modules 압축 해제 중..."
        tar xzf "$PACKAGE_DIR/frontend-dist/node_modules.tar.gz" -C "$INSTALL_DIR/frontend/" 2>/dev/null || \
            log_warn "node_modules 압축 해제 실패"
    fi
else
    log_warn "프론트엔드 디렉토리 없음, 스킵"
fi

# 5. 설정 파일 배치
echo ""
log_info "[5/5] 설정 파일 배치..."

# 환경 설정 파일
if [ -f "$PACKAGE_DIR/config/.env.example" ]; then
    if [ ! -f "$INSTALL_DIR/backend/.env" ]; then
        cp "$PACKAGE_DIR/config/.env.example" "$INSTALL_DIR/backend/.env"
        log_info ".env 생성됨 (수정 필요: $INSTALL_DIR/backend/.env)"
    else
        log_warn ".env 이미 존재 (스킵)"
    fi
fi

# Docker Compose 파일
if [ -f "$PACKAGE_DIR/config/docker-compose.monitoring.yml" ]; then
    cp "$PACKAGE_DIR/config/docker-compose.monitoring.yml" "$INSTALL_DIR/"
    log_info "docker-compose.monitoring.yml 복사 완료"
fi

if [ -f "$PACKAGE_DIR/config/docker-compose.ocr.yml" ]; then
    cp "$PACKAGE_DIR/config/docker-compose.ocr.yml" "$INSTALL_DIR/"
    log_info "docker-compose.ocr.yml 복사 완료"
fi

# 프롬프트 템플릿
if [ -d "$PACKAGE_DIR/config/prompts" ]; then
    cp -r "$PACKAGE_DIR/config/prompts" "$INSTALL_DIR/backend/"
    log_info "프롬프트 템플릿 복사 완료"
fi

# 임베딩 모델 (있으면)
if [ -d "$PACKAGE_DIR/models/embeddings" ]; then
    log_info "임베딩 모델 복사 중..."
    mkdir -p "$HOME/.cache/huggingface/hub"
    cp -r "$PACKAGE_DIR/models/embeddings/"* "$HOME/.cache/huggingface/hub/" 2>/dev/null || \
        log_warn "임베딩 모델 복사 실패"
fi

# systemd 서비스 파일 생성 (선택)
if command -v systemctl &> /dev/null; then
    log_info "systemd 서비스 파일 생성 중..."
    cat > "/tmp/flux-rag-backend.service" <<EOF
[Unit]
Description=Flux RAG Backend Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR/backend
Environment="PATH=$INSTALL_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$INSTALL_DIR/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    log_info "systemd 서비스 파일 생성됨: /tmp/flux-rag-backend.service"
    log_info "  설치: sudo cp /tmp/flux-rag-backend.service /etc/systemd/system/"
    log_info "  활성화: sudo systemctl enable flux-rag-backend.service"
    log_info "  시작: sudo systemctl start flux-rag-backend.service"
fi

# 권한 설정
log_info "권한 설정 중..."
chown -R $(whoami):$(id -gn) "$INSTALL_DIR" 2>/dev/null || true
chmod -R 755 "$INSTALL_DIR" 2>/dev/null || true

echo ""
echo "=========================================="
log_info "설치 완료!"
echo "=========================================="
echo ""
echo "설치 디렉토리: $INSTALL_DIR"
echo ""
echo "다음 단계:"
echo ""
echo "1. 환경 설정 수정:"
echo "   vi $INSTALL_DIR/backend/.env"
echo ""
echo "2. Docker 서비스 시작:"
echo "   cd $INSTALL_DIR"
echo "   docker compose -f docker-compose.monitoring.yml up -d"
echo ""
echo "3. 백엔드 실행:"
echo "   cd $INSTALL_DIR/backend"
echo "   source ../.venv/bin/activate"
echo "   uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4"
echo ""
echo "4. 접속 확인:"
echo "   curl http://localhost:8000/health"
echo ""
echo "5. 프론트엔드 (nginx 권장):"
echo "   nginx root: $INSTALL_DIR/frontend/"
echo ""
echo "=========================================="
echo ""
log_info "설치 스크립트 종료"
