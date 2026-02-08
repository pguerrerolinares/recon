FROM python:3.12-slim

WORKDIR /app

# Copy everything needed for install
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the package with memory support (non-editable, production mode)
RUN pip install --no-cache-dir ".[memory]"

WORKDIR /workspace

ENTRYPOINT ["recon"]
CMD ["--help"]
