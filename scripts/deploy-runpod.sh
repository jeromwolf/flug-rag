#!/bin/bash
# ============================================================
# RunPod 재배포 스크립트
# A100 (백엔드) + A40 (nginx 프록시) 구성
# ============================================================

set -e

# --- 설정 ---
A100_HOST="root@157.157.221.29"
A100_PORT="12689"
A40_HOST="root@194.68.245.208"
A40_PORT="22115"
SSH_KEY="~/.ssh/id_ed25519"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== flux-rag RunPod 배포 ==="
echo "로컬 경로: $LOCAL_DIR"

if [ -z "$A100_HOST" ] || [ -z "$A100_PORT" ]; then
  echo "❌ A100_HOST, A100_PORT를 스크립트 상단에 설정하세요."
  exit 1
fi

# --- 1. A100에 백엔드 배포 ---
echo ""
echo ">>> [1/5] A100에 백엔드 소스 전송..."
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='node_modules' \
  --exclude='*.pyc' --exclude='.git' --exclude='frontend/dist' \
  -e "ssh -o StrictHostKeyChecking=no -p $A100_PORT -i $SSH_KEY" \
  "$LOCAL_DIR/" "$A100_HOST:/workspace/flux-rag/"

# --- 2. A100에 데이터 복원 (milvus.db 등) ---
echo ""
echo ">>> [2/5] 데이터 파일 확인..."
ssh -o StrictHostKeyChecking=no -p "$A100_PORT" -i "$SSH_KEY" "$A100_HOST" bash <<'REMOTE'
cd /workspace/flux-rag/backend
if [ ! -f data/milvus.db ]; then
  echo "⚠️  milvus.db 없음. flux-rag-data.tar.gz에서 복원 시도..."
  if [ -f /workspace/flux-rag/flux-rag-data.tar.gz ]; then
    cd /workspace/flux-rag && tar xzf flux-rag-data.tar.gz
    echo "✅ 데이터 복원 완료"
  else
    echo "❌ flux-rag-data.tar.gz가 없습니다. 로컬에서 scp로 전송하세요:"
    echo "   scp -P $A100_PORT flux-rag-data.tar.gz $A100_HOST:/workspace/flux-rag/"
  fi
else
  echo "✅ milvus.db 존재 ($(du -h data/milvus.db | cut -f1))"
fi
REMOTE

# --- 3. A100 백엔드 시작 ---
echo ""
echo ">>> [3/5] A100 백엔드 시작..."
ssh -o StrictHostKeyChecking=no -p "$A100_PORT" -i "$SSH_KEY" "$A100_HOST" bash <<'REMOTE'
cd /workspace/flux-rag/backend

# .env 확인
if [ ! -f .env ]; then
  echo "⚠️  .env 없음. .env.example에서 복사합니다."
  cp .env.example .env
  echo "   ✏️  .env를 환경에 맞게 수정하세요!"
fi

# 의존성 설치
pip install -q -r requirements.txt 2>/dev/null

# 기존 프로세스 정리
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

# 백엔드 시작
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 \
  > /workspace/flux-rag/backend.log 2>&1 &
echo "✅ 백엔드 시작 (PID: $!)"
sleep 3
curl -s http://localhost:8000/health | head -1
REMOTE

# --- 4. 프론트엔드 빌드 & A40 배포 ---
echo ""
echo ">>> [4/5] 프론트엔드 빌드 & A40 배포..."
cd "$LOCAL_DIR/frontend"
npm run build 2>&1 | tail -3

scp -P "$A40_PORT" -i "$SSH_KEY" dist/index.html "$A40_HOST:/workspace/flux-rag/frontend/dist/index.html"
scp -P "$A40_PORT" -i "$SSH_KEY" dist/assets/* "$A40_HOST:/workspace/flux-rag/frontend/dist/assets/"
scp -rP "$A40_PORT" -i "$SSH_KEY" dist/guide/ "$A40_HOST:/workspace/flux-rag/frontend/dist/guide/"
echo "✅ A40 프론트엔드 배포 완료"

# --- 5. nginx 확인 ---
echo ""
echo ">>> [5/5] A40 nginx 확인..."
ssh -o StrictHostKeyChecking=no -p "$A40_PORT" -i "$SSH_KEY" "$A40_HOST" \
  "nginx -t 2>&1 && nginx -s reload 2>/dev/null; echo '✅ nginx OK'"

echo ""
echo "=== 배포 완료 ==="
echo "URL: https://7rzubyo9fsfmco-3000.proxy.runpod.net/"
echo "가이드: https://7rzubyo9fsfmco-3000.proxy.runpod.net/guide/demo.html"
