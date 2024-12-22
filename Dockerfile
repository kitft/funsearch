FROM docker.io/python:3.11.6


WORKDIR /workspace

# Install PDM and other essential tools
RUN pip install pdm

# Copy dependency files
COPY pyproject.toml README.md pdm.lock ./

# Install dependencies
ENV PATH="/workspace/.venv/bin:$PATH"
RUN pdm install --no-self
RUN pip install mistralai tensorboard  
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install matplotlib pandas
RUN pip install anthropic google-generativeai
RUN pip install openai>=1.2 
RUN pip install click
RUN pip install cloudpickle
RUN pip install scipy

# Create necessary subfolders in data directory if they don't exist
RUN mkdir -p ./data && \
    cd ./data && \
    mkdir -p scores graphs backups tensorboard_logs &&\
    cd ..

# Copy application code
COPY examples ./examples
COPY funsearch ./funsearch

# Install the application
RUN pip install --no-deps . && rm -r ./funsearch ./build


# Uncomment if needed - will instead be installed in the container when plotting commands are run
#RUN pip install pandas matplotlib
#RUN pip install llm-mistral

# Set Mistral API key
#CMD echo '{"mistral": "'$MISTRAL_API_KEY'"}' > $(llm keys path); /bin/bash
EXPOSE 6006

CMD ["bash"]

