#!/bin/bash
# =============================================================================
# RunPod GPU Pod Setup Script for flux-rag
# =============================================================================
# Usage:
#   1. Create RunPod Pod (A40 48GB recommended)
#   2. SSH into your RunPod Pod
#   3. Run: bash scripts/runpod_setup.sh
#
# Or one-liner from scratch:
#   git clone https://github.com/jeromwolf/flug-rag.git /workspace/flux-rag && \
#   bash /workspace/flux-rag/scripts/runpod_setup.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${CYAN}[$1]${NC} $2"; }

PROJECT_DIR="/workspace/flux-rag"
TOTAL_STEPS=10

echo ""
echo "============================================"
echo "  flux-rag RunPod Full Setup"
echo "  Backend + Frontend + LLM"
echo "============================================"
echo ""

# =============================================================================
# [1] System Info & GPU Detection
# =============================================================================
log_step "1/$TOTAL_STEPS" "System Info & GPU Detection"

uname -a
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h /workspace | tail -1 | awk '{print $4}') free"

# GPU detection and model selection
GPU_NAME="none"
GPU_VRAM_MB=0
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "none")
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
fi

echo "GPU: $GPU_NAME (${GPU_VRAM_MB}MB VRAM)"

# Auto-select model based on VRAM
if [ "$GPU_VRAM_MB" -ge 45000 ]; then
    # 48GB+ (A40, A6000, A100) → 32b fits well, 72b is tight
    OLLAMA_MODEL="qwen2.5:32b"
    LIGHT_MODEL="qwen2.5:14b"
    log_info "GPU >= 48GB: Using qwen2.5:32b (main) + qwen2.5:14b (light)"
elif [ "$GPU_VRAM_MB" -ge 22000 ]; then
    # 24GB (3090, 4090, A10G) → 14b
    OLLAMA_MODEL="qwen2.5:14b"
    LIGHT_MODEL="qwen2.5:7b"
    log_info "GPU >= 24GB: Using qwen2.5:14b (main) + qwen2.5:7b (light)"
elif [ "$GPU_VRAM_MB" -ge 10000 ]; then
    # 16GB (T4, A10) → 7b
    OLLAMA_MODEL="qwen2.5:7b"
    LIGHT_MODEL="qwen2.5:3b"
    log_info "GPU >= 16GB: Using qwen2.5:7b"
else
    OLLAMA_MODEL="qwen2.5:7b"
    LIGHT_MODEL="qwen2.5:3b"
    log_warn "Low/No GPU: Using qwen2.5:7b (CPU mode - slow)"
fi

# Allow override
if [ -n "$FLUX_MODEL" ]; then
    OLLAMA_MODEL="$FLUX_MODEL"
    log_info "Model override: $OLLAMA_MODEL"
fi
if [ -n "$FLUX_LIGHT_MODEL" ]; then
    LIGHT_MODEL="$FLUX_LIGHT_MODEL"
    log_info "Light model override: $LIGHT_MODEL"
fi

echo ""
echo "  Main model:  $OLLAMA_MODEL"
echo "  Light model: $LIGHT_MODEL"

# =============================================================================
# [2] System Dependencies
# =============================================================================
log_step "2/$TOTAL_STEPS" "Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq \
    git curl wget build-essential zstd \
    python3-venv python3-pip python3-dev \
    redis-server nginx \
    2>/dev/null || true

# Node.js (for frontend)
if ! command -v node &> /dev/null; then
    log_info "Installing Node.js 20 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>/dev/null
    apt-get install -y -qq nodejs 2>/dev/null || true
fi
echo "Node.js: $(node --version 2>/dev/null || echo 'not installed')"
echo "npm: $(npm --version 2>/dev/null || echo 'not installed')"

# =============================================================================
# [3] Clone or Update Repository
# =============================================================================
log_step "3/$TOTAL_STEPS" "Setting up project..."

if [ -d "$PROJECT_DIR/.git" ]; then
    log_info "Project exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull origin main || true
else
    log_info "Cloning repository..."
    git clone https://github.com/jeromwolf/flug-rag.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# =============================================================================
# [4] Python Environment
# =============================================================================
log_step "4/$TOTAL_STEPS" "Setting up Python environment..."

cd "$PROJECT_DIR/backend"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

log_info "Installing Python dependencies..."
pip install -q \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "python-multipart>=0.0.18" \
    "openai>=1.60.0" \
    "anthropic>=0.43.0" \
    "httpx>=0.28.0" \
    "sentence-transformers>=3.4.0" \
    "chromadb>=0.6.0" \
    "pymupdf>=1.25.0" \
    "pyhwp==0.1b15" \
    "python-docx>=1.1.0" \
    "openpyxl>=3.1.0" \
    "python-pptx>=1.0.0" \
    "rank-bm25>=0.2.2" \
    "flashrank>=0.2.0" \
    "langchain-text-splitters>=0.3.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.7.0" \
    "aiosqlite>=0.20.0" \
    "minio>=7.2.0" \
    "python-jose[cryptography]>=3.3.0" \
    "passlib[bcrypt]>=1.7.4" \
    "bcrypt>=4.2.0" \
    "apscheduler>=3.10.0" \
    "watchdog>=6.0.0" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0.0" \
    "numpy" \
    "sse-starlette>=2.2.0" \
    "psutil>=5.9" \
    "redis>=5.0.0" \
    "rouge-score>=0.1.2" \
    "kiwipiepy" \
    2>&1 | tail -5

log_info "Python packages installed."
python3 -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"

# =============================================================================
# [5] Install & Start Ollama (models stored on persistent /workspace volume)
# =============================================================================
log_step "5/$TOTAL_STEPS" "Installing Ollama..."

# Store models on /workspace so they survive Pod stop/start
export OLLAMA_MODELS="/workspace/ollama_models"
mkdir -p "$OLLAMA_MODELS"

if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama (kill existing first)
pkill -f "ollama serve" 2>/dev/null || true
sleep 1
OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve &>/dev/null &
sleep 3

# Verify Ollama is running
for i in $(seq 1 10); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log_info "Ollama is running."
        break
    fi
    sleep 2
done

# Pull models (skip if already downloaded on /workspace)
EXISTING_MODELS=$(ollama list 2>/dev/null | grep -c "$OLLAMA_MODEL" || echo "0")
if [ "$EXISTING_MODELS" -eq 0 ]; then
    log_info "Pulling $OLLAMA_MODEL (this may take a while)..."
    ollama pull "$OLLAMA_MODEL"
else
    log_info "$OLLAMA_MODEL already downloaded (persistent volume). Skipping."
fi

if [ "$LIGHT_MODEL" != "$OLLAMA_MODEL" ]; then
    EXISTING_LIGHT=$(ollama list 2>/dev/null | grep -c "$LIGHT_MODEL" || echo "0")
    if [ "$EXISTING_LIGHT" -eq 0 ]; then
        log_info "Pulling $LIGHT_MODEL..."
        ollama pull "$LIGHT_MODEL"
    else
        log_info "$LIGHT_MODEL already downloaded. Skipping."
    fi
fi

echo ""
echo "Installed models:"
ollama list

# =============================================================================
# [6] Redis
# =============================================================================
log_step "6/$TOTAL_STEPS" "Starting Redis..."

redis-server --daemonize yes --port 6379 2>/dev/null || true
redis-cli ping 2>/dev/null && log_info "Redis OK" || log_warn "Redis not available"

# =============================================================================
# [7] Environment Configuration
# =============================================================================
log_step "7/$TOTAL_STEPS" "Configuring environment..."

ENV_FILE="$PROJECT_DIR/backend/.env"

# Always regenerate .env with detected model
cat > "$ENV_FILE" << ENVEOF
# ========== RunPod Config (auto-generated) ==========
# GPU: $GPU_NAME (${GPU_VRAM_MB}MB)
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

# LLM
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=$OLLAMA_MODEL
OLLAMA_LIGHT_MODEL=$LIGHT_MODEL

# Vector DB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Auth (disabled for demo)
AUTH_ENABLED=false
JWT_SECRET_KEY=runpod-demo-jwt-secret-$(date +%s)

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

# OCR
UPSTAGE_API_KEY=
OCR_PROVIDER=cloud

# Monitoring
PROMETHEUS_ENABLED=false
ENVEOF

log_info "Created .env with model=$OLLAMA_MODEL"

# =============================================================================
# [8] Data Directories
# =============================================================================
log_step "8/$TOTAL_STEPS" "Setting up data directories..."

mkdir -p "$PROJECT_DIR/backend/data/uploads"
mkdir -p "$PROJECT_DIR/backend/data/chroma_db"

# Check if data was uploaded
CHROMA_FILES=$(find "$PROJECT_DIR/backend/data/chroma_db" -type f 2>/dev/null | wc -l)
UPLOAD_FILES=$(find "$PROJECT_DIR/backend/data/uploads" -type f 2>/dev/null | wc -l)

if [ "$CHROMA_FILES" -gt 0 ]; then
    log_info "ChromaDB data found: $CHROMA_FILES files"
else
    log_warn "ChromaDB data NOT found! Upload data before starting."
    log_warn "From local machine: scp flux-rag-data.tar.gz root@<POD_IP>:/workspace/"
    log_warn "Then run: cd /workspace && tar xzf flux-rag-data.tar.gz"
fi

if [ "$UPLOAD_FILES" -gt 0 ]; then
    log_info "Upload data found: $UPLOAD_FILES files"
fi

# =============================================================================
# [9] Frontend Build
# =============================================================================
log_step "9/$TOTAL_STEPS" "Building frontend..."

cd "$PROJECT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    log_info "Installing npm dependencies..."
    npm install --legacy-peer-deps 2>&1 | tail -3
fi

log_info "Building production bundle..."
npm run build 2>&1 | tail -5

if [ -d "dist" ]; then
    log_info "Frontend build successful: $(du -sh dist | awk '{print $1}')"
else
    log_error "Frontend build failed!"
fi

# =============================================================================
# [10] Nginx Configuration (Frontend serving + API proxy)
# =============================================================================
log_step "10/$TOTAL_STEPS" "Configuring nginx..."

cat > /etc/nginx/sites-available/flux-rag << 'NGINXEOF'
server {
    listen 3000;
    server_name _;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/javascript application/json application/javascript image/svg+xml;

    # Frontend static files
    root /workspace/flux-rag/frontend/dist;
    index index.html;

    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy → backend
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Health check proxy
    location /health {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
    }

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
NGINXEOF

# Save nginx config to persistent volume for restart recovery
cp /etc/nginx/sites-available/flux-rag "$PROJECT_DIR/scripts/nginx-flux-rag.conf"

# Enable site
ln -sf /etc/nginx/sites-available/flux-rag /etc/nginx/sites-enabled/flux-rag
rm -f /etc/nginx/sites-enabled/default 2>/dev/null

# Test and reload nginx
nginx -t 2>/dev/null && {
    systemctl restart nginx 2>/dev/null || nginx -s reload 2>/dev/null || nginx 2>/dev/null
    log_info "Nginx configured on port 3000"
} || {
    log_warn "Nginx config test failed, will need manual fix"
}

# =============================================================================
# Startup helper script
# =============================================================================
cat > "$PROJECT_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
# =============================================================================
# Start all flux-rag services
# Works for both fresh setup AND Pod restart after stop
# =============================================================================

echo ""
echo "============================================"
echo "  flux-rag Service Startup"
echo "============================================"
echo ""

# --- [0] Reinstall system packages if needed (after Pod stop/start) ---
if ! command -v nginx &> /dev/null || ! command -v redis-server &> /dev/null; then
    echo "[0/5] Reinstalling system packages (Pod was restarted)..."
    apt-get update -qq
    apt-get install -y -qq nginx redis-server curl 2>/dev/null || true
    echo "  System packages restored."
fi

if ! command -v ollama &> /dev/null; then
    echo "[0/5] Reinstalling Ollama binary..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama binary restored."
fi

# Restore nginx config if missing
if [ ! -f /etc/nginx/sites-available/flux-rag ]; then
    echo "[0/5] Restoring nginx config..."
    cp /workspace/flux-rag/scripts/nginx-flux-rag.conf /etc/nginx/sites-available/flux-rag 2>/dev/null || true
    ln -sf /etc/nginx/sites-available/flux-rag /etc/nginx/sites-enabled/flux-rag
    rm -f /etc/nginx/sites-enabled/default 2>/dev/null
fi

# --- [1] Ollama (models on persistent /workspace) ---
export OLLAMA_MODELS="/workspace/ollama_models"
if ! pgrep -x "ollama" > /dev/null; then
    echo "[1/5] Starting Ollama..."
    OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve &>/dev/null &
    sleep 3
    # Verify
    for i in $(seq 1 10); do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
else
    echo "[1/5] Ollama already running."
fi

# Show available models
echo "  Models: $(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ', ')"

# --- [2] Redis ---
if ! redis-cli ping &>/dev/null; then
    echo "[2/5] Starting Redis..."
    redis-server --daemonize yes --port 6379
else
    echo "[2/5] Redis already running."
fi

# --- [3] Backend ---
echo "[3/5] Starting backend..."
cd /workspace/flux-rag/backend
source .venv/bin/activate
pkill -f "uvicorn api.main" 2>/dev/null || true
sleep 1
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2 > /workspace/flux-rag/backend.log 2>&1 &
echo "  Backend PID: $! (log: /workspace/flux-rag/backend.log)"

# --- [4] Nginx (frontend) ---
echo "[4/5] Starting nginx..."
nginx -t 2>/dev/null && {
    nginx -s reload 2>/dev/null || nginx 2>/dev/null
} || {
    echo "  WARNING: nginx config issue, trying direct start..."
    nginx 2>/dev/null || true
}

# --- [5] Wait for backend warmup ---
echo "[5/5] Waiting for backend warmup (RAG chain init)..."
for i in $(seq 1 90); do
    if curl -s http://localhost:8000/health | grep -q "ok\|healthy" 2>/dev/null; then
        echo "  Backend ready! (${i}s)"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "  WARNING: Backend not responding after 90s. Check: tail -50 /workspace/flux-rag/backend.log"
    fi
    sleep 2
done

echo ""
echo "============================================"
echo "  All Services Running!"
echo "============================================"
echo "  Backend API:  http://localhost:8000"
echo "  Frontend:     http://localhost:3000"
echo "  API Docs:     http://localhost:8000/docs"
echo ""
echo "  RunPod Access:"
echo "    Dashboard > Pod > Connect > HTTP Port 3000"
echo ""
echo "  Test:"
echo "    bash /workspace/flux-rag/scripts/runpod_test.sh"
echo "============================================"
STARTEOF
chmod +x "$PROJECT_DIR/start.sh"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Python:   $(python3 --version)"
echo "  Node.js:  $(node --version 2>/dev/null || echo 'N/A')"
echo "  Ollama:   $(ollama --version 2>/dev/null || echo 'N/A')"
echo "  Redis:    $(redis-cli ping 2>/dev/null || echo 'not running')"
echo "  GPU:      $GPU_NAME (${GPU_VRAM_MB}MB)"
echo "  Model:    $OLLAMA_MODEL (main) / $LIGHT_MODEL (light)"
echo ""
echo "============================================"
echo "  Next Steps:"
echo "============================================"
echo ""

if [ "$CHROMA_FILES" -eq 0 ]; then
    echo "  1. UPLOAD DATA (required!):"
    echo "     From local machine:"
    echo "       bash scripts/pack_data.sh"
    echo "       scp flux-rag-data.tar.gz root@<POD_IP>:/workspace/"
    echo ""
    echo "     On RunPod:"
    echo "       cd /workspace && tar xzf flux-rag-data.tar.gz"
    echo ""
    echo "  2. Start services:"
    echo "       bash /workspace/flux-rag/start.sh"
else
    echo "  Data already uploaded! Start services:"
    echo "       bash /workspace/flux-rag/start.sh"
fi

echo ""
echo "  Access:"
echo "    RunPod Dashboard > Pod > Connect > HTTP Port 3000"
echo ""
echo "  Test:"
echo "    bash /workspace/flux-rag/scripts/runpod_test.sh"
echo ""
echo "============================================"
