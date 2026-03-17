FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY LICENSE /workspace/LICENSE
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY sample_data /workspace/sample_data
COPY entrypoint.sh /workspace/entrypoint.sh

RUN pip install --no-cache-dir ".[runtime]"

ENTRYPOINT ["/workspace/entrypoint.sh"]