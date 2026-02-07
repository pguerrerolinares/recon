FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/

# Re-install to register the package entry point
RUN pip install --no-cache-dir -e .

WORKDIR /workspace

ENTRYPOINT ["recon"]
CMD ["--help"]
