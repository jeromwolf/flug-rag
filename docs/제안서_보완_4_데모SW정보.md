# (4) 데모 S/W 정보

## 데모 시연 환경 규격

| 구분 | 항목 | 내용 |
|------|------|------|
| **LLM (생성형AI 언어모델)** | **모델명** | Qwen2.5-32B-Instruct (납품 시 Qwen2.5-72B-Instruct) |
| | **규격** | Base Model: Qwen2.5-32B (Alibaba Cloud, Apache 2.0), vLLM 서빙 |
| **RAG (검색증강생성)** | **모델명** | BGE-M3 + bge-reranker-v2-m3 + Milvus 2.6 |
| | **규격** | 임베딩: BAAI/bge-m3 (MIT) / 리랭커: bge-reranker-v2-m3 (Apache 2.0) / 벡터DB: Milvus 2.6.x (Apache 2.0) |

## 데모 시연 URL

| 항목 | 내용 |
|------|------|
| **시연 URL** | https://7rzubyo9fsfmco-3000.proxy.runpod.net/ |
| **계정 (평가위원)** | ID: evaluator / PW: Eval@2026! |
| **GPU 환경** | RunPod Cloud, NVIDIA A40 48GB |
