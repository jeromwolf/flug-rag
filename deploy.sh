#!/bin/bash

# flux-rag 배포 스크립트
# Docker Compose 또는 Kubernetes 환경에 빠르게 배포

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 사용법 출력
usage() {
    cat << EOF
flux-rag 배포 스크립트

사용법:
    $0 [COMMAND] [OPTIONS]

Commands:
    docker-build        Docker 이미지 빌드
    docker-up           Docker Compose로 실행
    docker-down         Docker Compose 중지 및 삭제
    docker-logs         Docker Compose 로그 확인
    k8s-build-push      Docker 이미지 빌드 후 레지스트리에 푸시
    k8s-deploy          Kubernetes에 배포
    k8s-delete          Kubernetes에서 삭제
    k8s-status          Kubernetes 배포 상태 확인

Options:
    -r, --registry      Docker 레지스트리 경로 (k8s-build-push용)
    -v, --version       이미지 버전 태그 (기본: latest)
    -h, --help          도움말 출력

예시:
    $0 docker-up
    $0 k8s-build-push -r ghcr.io/yourorg -v v1.0.0
    $0 k8s-deploy

EOF
}

# .env 파일 확인
check_env_file() {
    if [ ! -f .env ]; then
        print_warn ".env 파일이 없습니다."
        if [ -f .env.example ]; then
            print_info ".env.example을 복사합니다..."
            cp .env.example .env
            print_warn ".env 파일을 편집하여 API 키를 설정하세요!"
        else
            print_error ".env.example 파일도 없습니다."
            exit 1
        fi
    fi
}

# Docker 이미지 빌드
docker_build() {
    print_info "Docker 이미지 빌드 시작..."

    print_info "백엔드 이미지 빌드 중..."
    docker build -t flux-rag-backend:${VERSION} ./backend

    print_info "프론트엔드 이미지 빌드 중..."
    docker build -t flux-rag-frontend:${VERSION} ./frontend

    print_info "이미지 빌드 완료!"
    docker images | grep flux-rag
}

# Docker Compose 실행
docker_up() {
    check_env_file
    print_info "Docker Compose로 서비스 시작 중..."
    docker-compose up -d
    print_info "서비스 시작 완료!"
    print_info "프론트엔드: http://localhost:3000"
    print_info "백엔드 API: http://localhost:8000"
    print_info "API 문서: http://localhost:8000/docs"
}

# Docker Compose 중지
docker_down() {
    print_info "Docker Compose 서비스 중지 중..."
    docker-compose down
    print_info "서비스 중지 완료!"
}

# Docker Compose 로그
docker_logs() {
    docker-compose logs -f
}

# Kubernetes용 이미지 빌드 및 푸시
k8s_build_push() {
    if [ -z "$REGISTRY" ]; then
        print_error "레지스트리 경로를 지정하세요: -r ghcr.io/yourorg"
        exit 1
    fi

    print_info "Docker 이미지 빌드 및 푸시 시작..."

    # 백엔드
    print_info "백엔드 이미지 빌드 중..."
    docker build -t ${REGISTRY}/flux-rag-backend:${VERSION} ./backend
    print_info "백엔드 이미지 푸시 중..."
    docker push ${REGISTRY}/flux-rag-backend:${VERSION}

    # 프론트엔드
    print_info "프론트엔드 이미지 빌드 중..."
    docker build -t ${REGISTRY}/flux-rag-frontend:${VERSION} ./frontend
    print_info "프론트엔드 이미지 푸시 중..."
    docker push ${REGISTRY}/flux-rag-frontend:${VERSION}

    print_info "이미지 빌드 및 푸시 완료!"
    print_warn "Deployment YAML에서 이미지 경로를 다음으로 변경하세요:"
    print_warn "  - ${REGISTRY}/flux-rag-backend:${VERSION}"
    print_warn "  - ${REGISTRY}/flux-rag-frontend:${VERSION}"
}

# Kubernetes 배포
k8s_deploy() {
    print_info "Kubernetes에 배포 중..."

    # Secret이 이미 존재하는지 확인
    if kubectl get secret flux-rag-secret -n flux-rag >/dev/null 2>&1; then
        print_warn "Secret이 이미 존재합니다. 건너뜁니다."
    else
        print_warn "Secret을 생성해야 합니다."
        print_info "다음 명령으로 Secret을 생성하세요:"
        echo "kubectl create secret generic flux-rag-secret \\"
        echo "  --from-literal=VLLM_API_KEY='your-key' \\"
        echo "  --from-literal=OPENAI_API_KEY='your-key' \\"
        echo "  --from-literal=SECRET_KEY='your-secret' \\"
        echo "  --namespace=flux-rag"
        read -p "계속하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    kubectl apply -f k8s/
    print_info "Kubernetes 배포 완료!"
    print_info "배포 상태를 확인하려면: $0 k8s-status"
}

# Kubernetes 삭제
k8s_delete() {
    print_warn "Kubernetes에서 모든 리소스를 삭제합니다..."
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete -f k8s/
        print_info "삭제 완료!"
    else
        print_info "취소되었습니다."
    fi
}

# Kubernetes 상태 확인
k8s_status() {
    print_info "Kubernetes 배포 상태:"
    echo ""
    echo "=== Namespace ==="
    kubectl get namespace flux-rag
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n flux-rag
    echo ""
    echo "=== Services ==="
    kubectl get svc -n flux-rag
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n flux-rag
    echo ""
    echo "=== PVC ==="
    kubectl get pvc -n flux-rag
}

# 기본값
VERSION="latest"
REGISTRY=""

# 인자 파싱
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        docker-build|docker-up|docker-down|docker-logs)
            COMMAND=$1
            shift
            ;;
        k8s-build-push|k8s-deploy|k8s-delete|k8s-status)
            COMMAND=$1
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
done

# 명령 실행
case $COMMAND in
    docker-build)
        docker_build
        ;;
    docker-up)
        docker_up
        ;;
    docker-down)
        docker_down
        ;;
    docker-logs)
        docker_logs
        ;;
    k8s-build-push)
        k8s_build_push
        ;;
    k8s-deploy)
        k8s_deploy
        ;;
    k8s-delete)
        k8s_delete
        ;;
    k8s-status)
        k8s_status
        ;;
    *)
        print_error "명령을 지정하세요."
        usage
        exit 1
        ;;
esac
