FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest

# Set up Jupyter
RUN python -m ipykernel install --user --name=python3

# Create non-root user
RUN useradd -m -s /bin/bash developer
USER developer

# Expose Jupyter port
EXPOSE 8888

CMD ["bash"] 