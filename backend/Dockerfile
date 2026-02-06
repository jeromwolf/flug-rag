# flux-rag Backend Dockerfile
# Python 3.11 기반 FastAPI 애플리케이션

# 빌드 스테이지: Poetry를 사용한 의존성 설치
FROM python:3.11-slim AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (빌드 도구)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
ENV POETRY_VERSION=1.8.0
ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# 의존성 파일 복사
COPY pyproject.toml poetry.lock* ./

# 의존성을 requirements.txt로 내보내기
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# 런타임 스테이지: 최소 이미지로 실행 환경 구성
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 런타임 의존성 설치 (curl for health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 빌더 스테이지에서 requirements.txt 복사
COPY --from=builder /app/requirements.txt .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성 (볼륨 마운트 포인트)
RUN mkdir -p /app/data/chroma_db

# 비루트 유저 생성 및 권한 설정
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
