#!/usr/bin/env bash
# flux-rag 복구 스크립트
#
# 사용법:
#   bash scripts/backup/restore.sh <백업파일.tar.gz> [복구_대상_디렉토리]
#
# 예시:
#   bash scripts/backup/restore.sh /opt/flux-rag/backups/flux-rag-backup-20240101_020000.tar.gz
#
# RTO 목표: 2시간 이내
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "사용법: $0 <백업파일.tar.gz> [복구대상디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 /opt/flux-rag/backups/flux-rag-backup-20240101_020000.tar.gz"
    echo "  $0 /opt/flux-rag/backups/flux-rag-backup-20240101_020000.tar.gz /opt/flux-rag/backend"
    exit 1
fi

BACKUP_FILE="$1"
BACKEND_DIR="${2:-/opt/flux-rag/backend}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: 백업 파일을 찾을 수 없습니다: $BACKUP_FILE"
    exit 1
fi

echo "=== flux-rag 복구 시작: $(date) ==="
echo "백업 파일: $BACKUP_FILE"
echo "복구 대상: $BACKEND_DIR"
echo ""

# 확인
read -p "계속하시겠습니까? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소됨."
    exit 0
fi

# 임시 디렉토리에 압축 해제
TEMP_DIR=$(mktemp -d)
echo "[1/6] 백업 파일 압축 해제..."
tar xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# 백업 디렉토리 찾기 (첫 번째 하위 디렉토리)
BACKUP_CONTENT=$(ls -d "$TEMP_DIR"/*/ | head -1)
echo "  백업 내용: $BACKUP_CONTENT"

# 2. 서비스 중지 안내
echo ""
echo "[2/6] 서비스 중지..."
echo "  WARNING: 복구 전 서비스를 중지하세요:"
echo "    - uvicorn / gunicorn 프로세스 종료"
echo "    - docker compose down (필요 시)"
read -p "서비스를 중지했습니까? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "서비스를 먼저 중지한 후 다시 실행하세요."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 3. ChromaDB 복구
echo ""
echo "[3/6] ChromaDB 복구..."
if [ -f "$BACKUP_CONTENT/chroma_db.tar.gz" ]; then
    # 기존 ChromaDB 백업
    if [ -d "$BACKEND_DIR/data/chroma_db" ]; then
        mv "$BACKEND_DIR/data/chroma_db" "$BACKEND_DIR/data/chroma_db.pre-restore"
    fi
    tar xzf "$BACKUP_CONTENT/chroma_db.tar.gz" -C "$BACKEND_DIR/data/"
    echo "  ChromaDB 복구 완료"
else
    echo "  SKIP: ChromaDB 백업 없음"
fi

# 4. SQLite DB 복구
echo ""
echo "[4/6] SQLite 데이터베이스 복구..."
if [ -d "$BACKUP_CONTENT/databases" ]; then
    for db_file in "$BACKUP_CONTENT/databases/"*.db; do
        if [ -f "$db_file" ]; then
            DB_NAME=$(basename "$db_file")
            # 기존 DB 백업
            if [ -f "$BACKEND_DIR/data/$DB_NAME" ]; then
                cp "$BACKEND_DIR/data/$DB_NAME" "$BACKEND_DIR/data/${DB_NAME}.pre-restore"
            fi
            cp "$db_file" "$BACKEND_DIR/data/$DB_NAME"
            echo "  $DB_NAME 복구됨"
        fi
    done
else
    echo "  SKIP: 데이터베이스 백업 없음"
fi

# 5. 프롬프트 복구
echo ""
echo "[5/6] 프롬프트 복구..."
if [ -f "$BACKUP_CONTENT/prompts.tar.gz" ]; then
    tar xzf "$BACKUP_CONTENT/prompts.tar.gz" -C "$BACKEND_DIR/"
    echo "  프롬프트 복구 완료"
else
    echo "  SKIP: 프롬프트 백업 없음"
fi

# 6. 업로드 문서 복구
echo ""
echo "[6/6] 업로드 문서 복구..."
if [ -f "$BACKUP_CONTENT/uploads.tar.gz" ]; then
    tar xzf "$BACKUP_CONTENT/uploads.tar.gz" -C "$BACKEND_DIR/data/"
    echo "  업로드 문서 복구 완료"
else
    echo "  SKIP: 업로드 문서 백업 없음"
fi

# 정리
rm -rf "$TEMP_DIR"

echo ""
echo "=== 복구 완료: $(date) ==="
echo ""
echo "다음 단계:"
echo "  1. .env 설정 확인: cat $BACKEND_DIR/.env"
echo "  2. Docker 서비스 시작: docker compose up -d"
echo "  3. 백엔드 시작: uvicorn api.main:app --port 8000"
echo "  4. 헬스체크: curl http://localhost:8000/health"
echo ""
echo "이전 데이터 정리 (복구 확인 후):"
echo "  rm -rf $BACKEND_DIR/data/chroma_db.pre-restore"
echo "  rm -f $BACKEND_DIR/data/*.pre-restore"
