# flux-rag Frontend Dockerfile
# Node.js 빌드 후 nginx로 정적 파일 서빙

# 빌드 스테이지: Node.js 환경에서 React 애플리케이션 빌드
FROM node:18-alpine AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY package*.json ./

# 의존성 설치 (ci는 package-lock.json 기반 정확한 버전 설치)
RUN npm ci

# 소스 코드 복사
COPY . .

# 프로덕션 빌드 실행
RUN npm run build

# 런타임 스테이지: nginx로 정적 파일 서빙
FROM nginx:alpine

# 빌드된 정적 파일 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# 커스텀 nginx 설정 복사
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx 사용자 권한 설정
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# 포트 노출
EXPOSE 80

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# nginx 실행 (daemon off로 포그라운드 실행)
CMD ["nginx", "-g", "daemon off;"]
