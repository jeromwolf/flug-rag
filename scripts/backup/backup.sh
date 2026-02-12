#!/usr/bin/env bash
# flux-rag 백업 스크립트
#
# 백업 대상:
#   - ChromaDB persist 디렉토리
#   - SQLite 데이터베이스 (memory.db, audit.db, ethics.db 등)
#   - .env 설정 파일
#   - 프롬프트 YAML 파일
#   - 업로드 문서
#
# 사용법:
#   bash scripts/backup/backup.sh [백업_디렉토리]
#
# crontab 설정 (매일 02:00):
#   0 2 * * * /opt/flux-rag/scripts/backup/backup.sh /opt/flux-rag/backups >> /var/log/flux-rag-backup.log 2>&1
set -euo pipefail

# 설정
BACKEND_DIR="${BACKEND_DIR:-/opt/flux-rag/backend}"
BACKUP_BASE="${1:-/opt/flux-rag/backups}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE/$DATE"
RETAIN_DAYS="${RETAIN_DAYS:-30}"  # 보관 기간 (일)

echo "=== flux-rag 백업 시작: $(date) ==="
echo "소스: $BACKEND_DIR"
echo "대상: $BACKUP_DIR"

mkdir -p "$BACKUP_DIR"

# 1. ChromaDB
echo ""
echo "[1/5] ChromaDB 백업..."
CHROMA_DIR="$BACKEND_DIR/data/chroma_db"
if [ -d "$CHROMA_DIR" ]; then
    tar czf "$BACKUP_DIR/chroma_db.tar.gz" -C "$BACKEND_DIR/data" "chroma_db"
    echo "  ChromaDB 백업 완료: $(du -sh "$BACKUP_DIR/chroma_db.tar.gz" | cut -f1)"
else
    echo "  SKIP: ChromaDB 디렉토리 없음"
fi

# 2. SQLite 데이터베이스
echo ""
echo "[2/5] SQLite 데이터베이스 백업..."
mkdir -p "$BACKUP_DIR/databases"
for db_file in "$BACKEND_DIR/data/"*.db; do
    if [ -f "$db_file" ]; then
        DB_NAME=$(basename "$db_file")
        # SQLite online backup (safe during writes)
        sqlite3 "$db_file" ".backup '$BACKUP_DIR/databases/$DB_NAME'" 2>/dev/null || \
            cp "$db_file" "$BACKUP_DIR/databases/$DB_NAME"
        echo "  $DB_NAME: $(du -sh "$BACKUP_DIR/databases/$DB_NAME" | cut -f1)"
    fi
done

# 3. 설정 파일
echo ""
echo "[3/5] 설정 파일 백업..."
mkdir -p "$BACKUP_DIR/config"
[ -f "$BACKEND_DIR/.env" ] && cp "$BACKEND_DIR/.env" "$BACKUP_DIR/config/.env"
[ -f "$BACKEND_DIR/config/settings.py" ] && cp "$BACKEND_DIR/config/settings.py" "$BACKUP_DIR/config/"
echo "  설정 파일 백업 완료"

# 4. 프롬프트 YAML
echo ""
echo "[4/5] 프롬프트 백업..."
PROMPTS_DIR="$BACKEND_DIR/prompts"
if [ -d "$PROMPTS_DIR" ]; then
    tar czf "$BACKUP_DIR/prompts.tar.gz" -C "$BACKEND_DIR" "prompts"
    echo "  프롬프트 백업 완료"
else
    echo "  SKIP: 프롬프트 디렉토리 없음"
fi

# 5. 업로드 문서
echo ""
echo "[5/5] 업로드 문서 백업..."
UPLOADS_DIR="$BACKEND_DIR/data/uploads"
if [ -d "$UPLOADS_DIR" ]; then
    tar czf "$BACKUP_DIR/uploads.tar.gz" -C "$BACKEND_DIR/data" "uploads"
    echo "  업로드 문서 백업 완료: $(du -sh "$BACKUP_DIR/uploads.tar.gz" | cut -f1)"
else
    echo "  SKIP: 업로드 디렉토리 없음"
fi

# 백업 메타데이터
cat > "$BACKUP_DIR/backup-info.json" << EOF
{
    "date": "$DATE",
    "source": "$BACKEND_DIR",
    "hostname": "$(hostname)",
    "files": $(ls "$BACKUP_DIR" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip().split('\n')))" 2>/dev/null || echo "[]")
}
EOF

# 최종 압축
cd "$BACKUP_BASE"
tar czf "flux-rag-backup-$DATE.tar.gz" "$DATE/"
BACKUP_SIZE=$(du -sh "flux-rag-backup-$DATE.tar.gz" | cut -f1)
rm -rf "$DATE/"

# 오래된 백업 정리
echo ""
echo "오래된 백업 정리 ($RETAIN_DAYS일 이전)..."
find "$BACKUP_BASE" -name "flux-rag-backup-*.tar.gz" -mtime +$RETAIN_DAYS -delete -print 2>/dev/null | head -5

echo ""
echo "=== 백업 완료: $(date) ==="
echo "파일: $BACKUP_BASE/flux-rag-backup-$DATE.tar.gz ($BACKUP_SIZE)"
