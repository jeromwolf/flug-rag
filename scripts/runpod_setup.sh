#!/bin/bash
# =============================================================================
# RunPod GPU Pod Setup Script for flux-rag
# =============================================================================
# Usage:
#   1. SSH into your RunPod Pod
#   2. Copy this script or clone the repo
#   3. Run: bash scripts/runpod_setup.sh
# =============================================================================

set -e

echo "============================================"
echo "  flux-rag RunPod Setup"
echo "============================================"

# --- System info ---
echo ""
echo "[1/8] System Info"
echo "---"
uname -a
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: No GPU detected"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo ""

# --- Install system dependencies ---
echo "[2/8] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
    git curl wget build-essential \
    python3.11 python3.11-venv python3-pip \
    redis-server \
    2>/dev/null || true

# Ensure python3 points to 3.11+
python3 --version

# --- Clone or update repo ---
echo ""
echo "[3/8] Setting up project..."
PROJECT_DIR="/workspace/flux-rag"

if [ -d "$PROJECT_DIR" ]; then
    echo "Project exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull origin main || true
else
    echo "Cloning repository..."
    echo ">>> You need to clone the repo manually or use rsync <<<"
    echo ">>> Option A: git clone <your-repo-url> $PROJECT_DIR"
    echo ">>> Option B: From local machine: rsync -avz --exclude='.venv' --exclude='data/chroma_db' -e 'ssh -i ~/.ssh/id_ed25519' ./backend/ 5nwlnv1i1rwafk-64411f61@ssh.runpod.io:/workspace/flux-rag/backend/"
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR/backend"

# --- Python virtual environment ---
echo ""
echo "[4/8] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# Install poetry and dependencies
pip install poetry -q
poetry config virtualenvs.create false
poetry install --no-interaction --no-ansi 2>&1 | tail -5

# Install Korean tokenizer for BM25
pip install kiwipiepy -q

echo "Python packages installed."
python3 -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"

# --- Install Ollama ---
echo ""
echo "[5/8] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama in background
echo "Starting Ollama server..."
ollama serve &>/dev/null &
sleep 3

# Pull models
echo "Pulling qwen2.5:14b (this may take a while)..."
ollama pull qwen2.5:14b

echo "Pulling embedding model info check..."
ollama list

# --- Redis ---
echo ""
echo "[6/8] Starting Redis..."
redis-server --daemonize yes --port 6379 2>/dev/null || true
redis-cli ping || echo "WARNING: Redis not available"

# --- Environment configuration ---
echo ""
echo "[7/8] Configuring environment..."
ENV_FILE="$PROJECT_DIR/backend/.env"

if [ ! -f "$ENV_FILE" ]; then
cat > "$ENV_FILE" << 'ENVEOF'
# ========== RunPod Production Config ==========

# LLM
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Auth
AUTH_ENABLED=false
JWT_SECRET_KEY=change-me-in-production-jwt-secret

# Cache (Redis)
CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379/0

# Performance
BATCH_INFERENCE_ENABLED=true
BATCH_SIZE=16
BATCH_MAX_WAIT_MS=50

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Chunking
CHUNK_STRATEGY=recursive
CHUNK_SIZE=800
CHUNK_OVERLAP=80

# RAG Tuning
RETRIEVAL_TOP_K=30
RERANK_TOP_N=7
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
RETRIEVAL_SCORE_THRESHOLD=0.0
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.1
USE_RERANK=true

# Advanced RAG
MULTI_QUERY_ENABLED=false
SELF_RAG_ENABLED=false
AGENTIC_RAG_ENABLED=false
QUERY_EXPANSION_ENABLED=false

# OCR (set your key)
UPSTAGE_API_KEY=
OCR_PROVIDER=cloud

# Monitoring
PROMETHEUS_ENABLED=false
ENVEOF
    echo "Created .env file. Edit it to set API keys."
else
    echo ".env already exists, skipping."
fi

# --- Create data directories ---
echo ""
echo "[8/8] Creating data directories..."
mkdir -p "$PROJECT_DIR/backend/data/uploads"
mkdir -p "$PROJECT_DIR/backend/data/chroma_db"

# --- Verify installation ---
echo ""
echo "============================================"
echo "  Setup Complete! Verification:"
echo "============================================"
echo ""
echo "Python:  $(python3 --version)"
echo "Ollama:  $(ollama --version 2>/dev/null || echo 'not found')"
echo "Redis:   $(redis-cli ping 2>/dev/null || echo 'not running')"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""
echo "============================================"
echo "  Next Steps:"
echo "============================================"
echo ""
echo "  1. Upload your data to: $PROJECT_DIR/backend/data/uploads/"
echo "  2. Edit .env if needed:  nano $PROJECT_DIR/backend/.env"
echo "  3. Start the server:"
echo "     cd $PROJECT_DIR/backend"
echo "     source .venv/bin/activate"
echo "     uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4"
echo ""
echo "  4. Access from outside RunPod:"
echo "     - RunPod dashboard → Pod → Connect → HTTP Port 8000"
echo "     - Or use SSH port forwarding:"
echo "       ssh -L 8000:localhost:8000 -i ~/.ssh/id_ed25519 5nwlnv1i1rwafk-64411f61@ssh.runpod.io"
echo ""
echo "============================================"
