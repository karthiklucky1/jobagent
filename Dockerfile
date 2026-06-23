FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser and its OS-level system dependencies
RUN playwright install --with-deps chromium

COPY . .

CMD ["sh", "-c", "exec uvicorn app.api.server:app --host 0.0.0.0 --port ${PORT:-8080}"]

