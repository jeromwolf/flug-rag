# RunPod 환경 셋업 가이드 (2026-03-09)

## 아키텍처 개요

```
평가단 브라우저
    ↓
7rzubyo9fsfmco-3000.proxy.runpod.net (A40 pod)
    ↓ nginx port 3000
    ├── 정적 파일 (frontend/dist) → A40 로컬 서빙
    └── /api, /health → proxy_pass → ju27h6ia4huffu-8000.proxy.runpod.net (A100 pod)
                                          ↓
                                     A100 nginx port 8001 → uvicorn port 8000
                                     A100 vLLM port 8002
```

**핵심**: 프론트엔드는 A40에서 서빙, 백엔드 API만 A100으로 프록시

## A40 Pod (프록시 + 프론트엔드)

- **Pod ID**: `7rzubyo9fsfmco` (URL 변경 불가 — 서식7에 제출됨)
- **SSH**: `ssh -o StrictHostKeyChecking=no -p 22115 root@194.68.245.208`
- **GPU**: A40 (사용 안 함, 프록시 전용)
- **비용**: $0.42/hr

### nginx 설정 위치
- `/etc/nginx/sites-enabled/proxy`

### nginx 설정 내용
```nginx
server {
    listen 3000;
    server_name _;

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/javascript application/json application/javascript image/svg+xml;

    root /workspace/flux-rag/frontend/dist;
    index index.html;

    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /api {
        proxy_pass https://ju27h6ia4huffu-8000.proxy.runpod.net;
        proxy_http_version 1.1;
        proxy_set_header Host ju27h6ia4huffu-8000.proxy.runpod.net;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_ssl_server_name on;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 100M;
    }

    location /health {
        proxy_pass https://ju27h6ia4huffu-8000.proxy.runpod.net;
        proxy_http_version 1.1;
        proxy_set_header Host ju27h6ia4huffu-8000.proxy.runpod.net;
        proxy_ssl_server_name on;
    }

    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires 0;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
```

### 프론트엔드 배포 방법
```bash
# 로컬에서 빌드
cd frontend && npm run build

# A40에 배포 (프론트엔드 서빙 서버)
ssh -o StrictHostKeyChecking=no -p 22115 root@194.68.245.208 "rm -rf /workspace/flux-rag/frontend/dist/assets/*"
scp -o StrictHostKeyChecking=no -P 22115 -r dist/* root@194.68.245.208:/workspace/flux-rag/frontend/dist/
ssh -o StrictHostKeyChecking=no -p 22115 root@194.68.245.208 "nginx -s reload"

# A100에도 배포 (백업용)
ssh -o StrictHostKeyChecking=no -p 12689 root@157.157.221.29 "rm -rf /workspace/flux-rag/frontend/dist/assets/*"
scp -o StrictHostKeyChecking=no -P 12689 -r dist/* root@157.157.221.29:/workspace/flux-rag/frontend/dist/
```

## A100 Pod (백엔드 + vLLM)

- **Pod ID**: `ju27h6ia4huffu`
- **SSH**: `ssh -o StrictHostKeyChecking=no -p 12689 root@157.157.221.29`
- **GPU**: A100 SXM 80GB
- **비용**: $1.49/hr

### 프로세스 구조
| 프로세스 | 포트 | 설명 |
|---------|------|------|
| vLLM | 8002 | Qwen2.5-32B-Instruct-AWQ |
| uvicorn (FastAPI) | 8000 | 백엔드 API |
| nginx | 3001 | 프론트엔드 서빙 (로컬 접속용) |
| nginx | 8001 | 백엔드 프록시 (RunPod 외부 노출) |

### vLLM 시작 명령
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen2.5-32B-Instruct-AWQ \
  --quantization awq_marlin \
  --port 8002 --host 0.0.0.0 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  --dtype half \
  --served-model-name qwen2.5-32b-instruct-awq \
  --enable-prefix-caching
```

### 백엔드 시작 명령
```bash
cd /workspace/flux-rag/backend
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### .env 핵심 설정 (`/workspace/flux-rag/backend/.env`)
```
DEFAULT_LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8002/v1
VECTORSTORE_TYPE=milvus_lite
MILVUS_STORE_URI=./data/milvus.db
AUTH_ENABLED=true
CONTEXT_MAX_CHUNKS=7
LLM_MAX_TOKENS=512
USE_RERANK=false
FEW_SHOT_MAX_EXAMPLES=2
CONFIDENCE_LOW=0.3
CONFIDENCE_HIGH=0.8
```

### nginx 설정 (`/etc/nginx/nginx.conf`)
```nginx
events { worker_connections 2048; }

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    client_max_body_size 1024M;
    sendfile on;

    server {
        listen 3001;
        root /workspace/flux-rag/frontend/dist;
        index index.html;

        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        location /api {
            proxy_pass http://127.0.0.1:8000;
            # ... SSE 지원 설정 ...
        }

        location = /index.html {
            add_header Cache-Control "no-cache, no-store, must-revalidate";
        }

        location / {
            try_files $uri $uri/ /index.html;
        }
    }

    server {
        listen 8001;
        location / {
            proxy_pass http://localhost:8000;
            # ... SSE 지원 설정 ...
        }
    }
}
```

### 백엔드 코드 수정 사항 (A100 직접 수정, git에 미반영)
1. **chain.py**: CJK(중국어) 실시간 필터링 — `_RE_STRIP_CJK` regex로 스트리밍 토큰에서 중국어 제거
2. **prompts/system.yaml**: 복수 조문 연계 + 부정 테스트 대응 규칙 추가

## 사용자 계정

| 역할 | 사용자명 | 비밀번호 | 용도 |
|------|---------|---------|------|
| Admin | admin | admin123 | 관리자 |
| Evaluator | evaluator | Eval@2026! | 서식7 평가단 |
| Group evaluators | KGT-G01~G03 | gggg | 그룹 평가 |

## 배포 체크리스트

### 프론트엔드 변경 시
1. 로컬에서 `npm run build`
2. **A40**에 scp 전송 + nginx reload (사용자가 접속하는 서버)
3. A100에도 scp 전송 (백업)
4. 브라우저 Ctrl+Shift+R 강력 새로고침

### 백엔드 변경 시
1. A100에 scp 전송 또는 SSH로 직접 수정
2. `pkill -f uvicorn; sleep 2; cd /workspace/flux-rag/backend && source .venv/bin/activate && nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &`

### 시스템 프롬프트 변경 시
1. A100의 `/workspace/flux-rag/backend/prompts/system.yaml` 직접 수정
2. 백엔드 재시작 필요 없음 (런타임에 파일 읽음)

## 주의사항
- A40 Pod ID `7rzubyo9fsfmco`는 서식7 URL이므로 절대 변경/삭제 불가
- A100 재시작 시 vLLM + uvicorn + nginx 모두 수동 시작 필요
- `/workspace/` 경로만 Pod 재시작 후 보존됨 (다른 경로는 초기화)
- 프론트엔드 배포 시 A40 + A100 양쪽 모두 배포 필수
