FROM docker.io/python:3.11.6

WORKDIR /workspace

# Copy build and dependency files first for better layer caching
COPY pyproject.toml README.md ./
COPY requirements*.txt ./

# Install dependencies using either uv or pip based on build arg
# Build with: docker build --build-arg USE_UV=true -t funsearch .
ARG USE_UV=false
RUN if [ "$USE_UV" = "true" ]; then \
        pip install --upgrade pip && \
        pip install uv && \
        uv pip install --system -r requirements.txt; \
    else \
        pip install --upgrade pip && \
        pip install -r requirements.txt; \
    fi

# Create necessary subfolders in data directory  
RUN mkdir -p ./data && \
    cd ./data && \
    mkdir -p scores graphs backups && \
    cd ..

# Copy application code
COPY examples ./examples
COPY funsearch ./funsearch

# Install the application
RUN if [ "$USE_UV" = "true" ]; then \
        uv pip install --system --no-deps .; \
    else \
        pip install --no-deps .; \
    fi && \
    rm -r ./funsearch ./build

CMD ["bash"]
