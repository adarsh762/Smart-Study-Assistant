# ── Stage 1: Build ──
FROM python:3.11-slim AS builder

WORKDIR /app

# Install CPU-only PyTorch first (saves ~1.5 GB vs full CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source code
COPY . .

# Create data and index directories
RUN mkdir -p /app/data /app/index

# HF Spaces expects port 7860
EXPOSE 7860

# Set production defaults
ENV ENV=production
ENV HOST=0.0.0.0
ENV PORT=7860

CMD ["python", "app.py"]
