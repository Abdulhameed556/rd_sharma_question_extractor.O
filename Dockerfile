# RD Sharma Question Extractor - Docker Setup
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the project in development mode
RUN pip install -e .

# Set Python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create necessary directories
RUN mkdir -p /app/outputs /app/data/cache /app/logs

# Expose Jupyter port
EXPOSE 8888

# Set default command to start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"] 