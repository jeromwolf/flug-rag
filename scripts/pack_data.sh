#!/bin/bash
# =============================================================================
# Pack flux-rag data for RunPod transfer
# =============================================================================
# Run this on your LOCAL machine to create a data archive.
# Then scp/rsync it to RunPod.
#
# Usage:
#   cd /path/to/flux-rag
#   bash scripts/pack_data.sh
#   scp flux-rag-data.tar.gz root@<RUNPOD_IP>:/workspace/
#
# On RunPod:
#   cd /workspace && tar xzf flux-rag-data.tar.gz
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"

echo ""
echo "============================================"
echo "  flux-rag Data Packer"
echo "============================================"
echo ""

# Check data exists
if [ ! -d "$BACKEND_DIR/data/chroma_db" ]; then
    echo -e "${RED}[ERROR]${NC} ChromaDB data not found at $BACKEND_DIR/data/chroma_db"
    exit 1
fi

echo "Scanning data..."
echo ""

CHROMA_SIZE=$(du -sh "$BACKEND_DIR/data/chroma_db" 2>/dev/null | awk '{print $1}')
UPLOADS_SIZE=$(du -sh "$BACKEND_DIR/data/uploads" 2>/dev/null | awk '{print $1}')
echo "  ChromaDB:  $CHROMA_SIZE"
echo "  Uploads:   $UPLOADS_SIZE"

# Check for SQLite DBs (memory, audit, users)
DB_FILES=""
for db in memory.db audit.db users.db abuse_detector.db; do
    if [ -f "$BACKEND_DIR/data/$db" ]; then
        DB_SIZE=$(du -sh "$BACKEND_DIR/data/$db" | awk '{print $1}')
        echo "  $db:   $DB_SIZE"
        DB_FILES="$DB_FILES backend/data/$db"
    fi
done

echo ""

# Create archive
OUTPUT_FILE="$PROJECT_DIR/flux-rag-data.tar.gz"
echo -e "${GREEN}[INFO]${NC} Creating archive..."

cd "$PROJECT_DIR"

# Build file list
TAR_PATHS="backend/data/chroma_db"

if [ -d "$BACKEND_DIR/data/uploads" ]; then
    TAR_PATHS="$TAR_PATHS backend/data/uploads"
fi

# Add DB files
TAR_PATHS="$TAR_PATHS $DB_FILES"

# Add sample dataset if exists
if [ -d "$BACKEND_DIR/data/sample_dataset" ]; then
    TAR_PATHS="$TAR_PATHS backend/data/sample_dataset"
    echo "  + sample_dataset included"
fi

# Create tar.gz
tar czf "$OUTPUT_FILE" $TAR_PATHS 2>/dev/null

ARCHIVE_SIZE=$(du -sh "$OUTPUT_FILE" | awk '{print $1}')

echo ""
echo "============================================"
echo -e "  ${GREEN}Archive created!${NC}"
echo "============================================"
echo ""
echo "  File: $OUTPUT_FILE"
echo "  Size: $ARCHIVE_SIZE"
echo ""
echo "  Transfer to RunPod:"
echo "    scp $OUTPUT_FILE root@<POD_IP>:/workspace/"
echo ""
echo "  On RunPod, extract:"
echo "    cd /workspace/flux-rag && tar xzf /workspace/flux-rag-data.tar.gz"
echo ""
echo "============================================"
