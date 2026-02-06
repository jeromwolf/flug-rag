# flux-rag Kubernetes 배포 가이드

이 디렉토리는 flux-rag 프로젝트의 Kubernetes 배포 설정을 포함합니다.

## 파일 구조

```
k8s/
├── namespace.yaml              # 네임스페이스 정의
├── configmap.yaml              # 비민감 환경 변수
├── secret.yaml                 # 민감 환경 변수 (API 키 등)
├── pvc.yaml                    # 데이터 영속성 볼륨
├── backend-deployment.yaml     # 백엔드 배포
├── backend-service.yaml        # 백엔드 서비스
├── frontend-deployment.yaml    # 프론트엔드 배포
├── frontend-service.yaml       # 프론트엔드 서비스
└── ingress.yaml                # 외부 접근 라우팅
```

## 사전 요구사항

1. **Kubernetes 클러스터** (v1.20 이상)
   - Minikube, Kind, EKS, GKE, AKS 등

2. **kubectl 설치** 및 클러스터 접근 설정

3. **Ingress Controller** (nginx 권장)
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
   ```

4. **Docker 이미지 빌드 및 푸시**
   ```bash
   # 백엔드 이미지 빌드
   cd backend
   docker build -t your-registry/flux-rag-backend:v1.0.0 .
   docker push your-registry/flux-rag-backend:v1.0.0

   # 프론트엔드 이미지 빌드
   cd frontend
   docker build -t your-registry/flux-rag-frontend:v1.0.0 .
   docker push your-registry/flux-rag-frontend:v1.0.0
   ```

## 배포 단계

### 1. Secret 설정 (중요!)

`secret.yaml`의 플레이스홀더 값을 실제 값으로 교체하세요:

```bash
# API 키를 base64로 인코딩
echo -n 'your-openai-api-key' | base64
echo -n 'your-anthropic-api-key' | base64
echo -n 'your-vllm-api-key' | base64

# 또는 파일로 Secret 생성 (권장)
kubectl create secret generic flux-rag-secret \
  --from-literal=VLLM_API_KEY='your-vllm-key' \
  --from-literal=OPENAI_API_KEY='your-openai-key' \
  --from-literal=ANTHROPIC_API_KEY='your-anthropic-key' \
  --from-literal=UPSTAGE_API_KEY='your-upstage-key' \
  --from-literal=SECRET_KEY='your-jwt-secret' \
  --namespace=flux-rag
```

### 2. 이미지 경로 수정

Deployment 파일에서 이미지 경로를 실제 레지스트리 경로로 변경:

```yaml
# backend-deployment.yaml
image: your-registry/flux-rag-backend:v1.0.0

# frontend-deployment.yaml
image: your-registry/flux-rag-frontend:v1.0.0
```

### 3. Ingress 도메인 설정

`ingress.yaml`의 호스트를 실제 도메인으로 변경:

```yaml
spec:
  rules:
  - host: flux-rag.example.com  # 실제 도메인으로 변경
```

### 4. 리소스 배포

```bash
# 전체 배포 (순서대로)
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml  # 또는 위의 create secret 명령 사용
kubectl apply -f pvc.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f frontend-service.yaml
kubectl apply -f ingress.yaml

# 또는 한 번에 배포
kubectl apply -f k8s/
```

### 5. 배포 확인

```bash
# Pod 상태 확인
kubectl get pods -n flux-rag

# 서비스 확인
kubectl get svc -n flux-rag

# Ingress 확인
kubectl get ingress -n flux-rag

# 로그 확인
kubectl logs -n flux-rag -l component=backend --tail=100 -f
kubectl logs -n flux-rag -l component=frontend --tail=100 -f
```

## GPU 지원 (옵션)

sentence-transformers 및 vLLM 추론 가속화를 위해 GPU를 사용하려면:

1. **NVIDIA GPU Operator 설치**
   ```bash
   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm repo update
   helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator
   ```

2. **backend-deployment.yaml에서 GPU 설정 주석 해제**
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 1

   nodeSelector:
     accelerator: nvidia-gpu

   tolerations:
   - key: nvidia.com/gpu
     operator: Exists
     effect: NoSchedule
   ```

## 스케일링

```bash
# 백엔드 Pod 수 조정
kubectl scale deployment flux-rag-backend --replicas=3 -n flux-rag

# 프론트엔드 Pod 수 조정
kubectl scale deployment flux-rag-frontend --replicas=3 -n flux-rag

# HPA (Horizontal Pod Autoscaler) 설정
kubectl autoscale deployment flux-rag-backend \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n flux-rag
```

## 업데이트

```bash
# 이미지 업데이트
kubectl set image deployment/flux-rag-backend \
  backend=your-registry/flux-rag-backend:v1.1.0 \
  -n flux-rag

# 롤아웃 상태 확인
kubectl rollout status deployment/flux-rag-backend -n flux-rag

# 롤백 (문제 발생 시)
kubectl rollout undo deployment/flux-rag-backend -n flux-rag
```

## 모니터링

```bash
# 리소스 사용량 확인
kubectl top pods -n flux-rag
kubectl top nodes

# 이벤트 확인
kubectl get events -n flux-rag --sort-by='.lastTimestamp'
```

## 삭제

```bash
# 전체 리소스 삭제
kubectl delete namespace flux-rag

# 또는 개별 삭제
kubectl delete -f k8s/
```

## 트러블슈팅

### Pod이 Pending 상태인 경우

```bash
kubectl describe pod <pod-name> -n flux-rag
```

- **PVC 문제**: StorageClass 확인 및 수정
- **리소스 부족**: 노드 리소스 확인 및 스케일 다운
- **이미지 Pull 실패**: 이미지 경로 및 레지스트리 인증 확인

### 백엔드 연결 실패

```bash
# 서비스 엔드포인트 확인
kubectl get endpoints -n flux-rag

# 네트워크 정책 확인
kubectl get networkpolicies -n flux-rag
```

### Ingress 접근 불가

```bash
# Ingress Controller 상태 확인
kubectl get pods -n ingress-nginx

# Ingress 이벤트 확인
kubectl describe ingress flux-rag-ingress -n flux-rag
```

## 보안 권장사항

1. **Secret 관리**: Sealed Secrets, External Secrets Operator 사용
2. **네트워크 정책**: Pod 간 통신 제한
3. **RBAC**: 최소 권한 원칙 적용
4. **이미지 스캔**: Trivy, Clair 등으로 취약점 검사
5. **TLS 인증서**: cert-manager로 자동 발급 및 갱신

## 참고 자료

- [Kubernetes 공식 문서](https://kubernetes.io/docs/)
- [nginx Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
