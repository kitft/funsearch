FROM docker.io/python:3.11.6

WORKDIR /workspace

# Use PDM to keep track of exact versions of dependencies
RUN pip install pdm
COPY pyproject.toml README.md pdm.lock ./
# install dependencies first. PDM also creates a /workspace/.venv here.
ENV PATH="/workspace/.venv/bin:$PATH"
RUN pdm install  --no-self
COPY examples ./examples
COPY funsearch ./funsearch

RUN pip install --no-deps . && rm -r ./funsearch ./build
RUN pip install llm-mistral
# Pget Mistral API key
ARG MISTRAL_API_KEY
ENV MISTRAL_API_KEY="$MISTRAL_API_KEY"

# Set Mistral API key
#RUN llm keys path
RUN if [ -z "$MISTRAL_API_KEY" ]; then echo "Error: MISTRAL_API_KEY is not set. Please provide it when building the image. via --build-arg MISTRAL_API_KEY=<your_key>"; exit 1; fi
RUN echo '{"mistral": "'$MISTRAL_API_KEY'"}'>  $(llm keys path)

CMD /bin/bash
