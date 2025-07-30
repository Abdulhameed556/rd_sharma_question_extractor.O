# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the RD Sharma Question Extractor in various environments.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet access for API calls

### Software Dependencies
- **Git**: For repository cloning
- **pip**: Python package manager
- **conda**: Alternative package manager (optional)

## Installation Methods

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/rd-sharma-extractor/question-extractor.git
cd question-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
python scripts/setup_environment.py
```

### Method 2: Conda Installation

```bash
# Clone the repository
git clone https://github.com/rd-sharma-extractor/question-extractor.git
cd question-extractor

# Create conda environment
conda env create -f environment.yml
conda activate rd-sharma-extractor

# Setup environment
python scripts/setup_environment.py
```

### Method 3: Development Installation

```bash
# Clone the repository
git clone https://github.com/rd-sharma-extractor/question-extractor.git
cd question-extractor

# Install in development mode
pip install -e ".[dev]"

# Setup environment
python scripts/setup_environment.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama-4-maverick-17b

# Document Processing
PDF_PATH=data/rd_sharma_complete.pdf
DOCUMENT_INDEX_PATH=data/document_index.json

# Output Configuration
OUTPUT_DIR=outputs
LATEX_OUTPUT_DIR=outputs/latex_files
JSON_OUTPUT_DIR=outputs/extracted_questions

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=outputs/logs/extraction.log
ERROR_LOG_FILE=outputs/logs/error.log
PERFORMANCE_LOG_FILE=outputs/logs/performance.log

# Performance Configuration
BATCH_SIZE=10
MAX_WORKERS=4
TIMEOUT_SECONDS=30
RETRY_ATTEMPTS=3
```

### API Key Setup

1. **Get Groq API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Create an account and generate an API key
   - Add the key to your `.env` file

2. **Verify API Access**:
   ```bash
   python src/main.py info
   ```

## Data Preparation

### PDF Document Setup

1. **Download RD Sharma PDF**:
   - Download from the provided Google Drive link
   - Place as `data/rd_sharma_complete.pdf`

2. **Verify PDF**:
   ```bash
   python scripts/test_ocr_quality.py data/rd_sharma_complete.pdf
   ```

3. **Build Document Index**:
   ```bash
   python scripts/build_index.py data/rd_sharma_complete.pdf
   ```

## Deployment Environments

### Local Development

```bash
# Run in development mode
python src/main.py extract --chapter 30 --topic 30.3 --debug

# Run tests
pytest tests/ -v

# Check code quality
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Production Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Setup production configuration
export LOG_LEVEL=WARNING
export DEBUG=False

# Run extraction
python src/main.py extract --chapter 30 --topic 30.3
```

### Docker Deployment (Future)

```dockerfile
# Dockerfile (future implementation)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/setup_environment.py

CMD ["python", "src/main.py", "extract", "--chapter", "30", "--topic", "30.3"]
```

## Monitoring and Logging

### Log Configuration

The system uses structured logging with multiple levels:

```python
# Log levels
DEBUG    # Detailed debugging information
INFO     # General information
WARNING  # Warning messages
ERROR    # Error messages
CRITICAL # Critical errors
```

### Performance Monitoring

```bash
# Monitor performance logs
tail -f outputs/logs/performance.log

# Check extraction logs
tail -f outputs/logs/extraction.log

# Monitor errors
tail -f outputs/logs/error.log
```

### Health Checks

```bash
# Check system health
python src/main.py info

# Test OCR quality
python scripts/test_ocr_quality.py data/rd_sharma_complete.pdf

# Validate LaTeX output
python src/main.py validate --file outputs/latex_files/chapter_30_3.tex
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```bash
   # Check environment variable
   echo $GROQ_API_KEY
   
   # Verify .env file
   cat .env
   ```

2. **PDF Not Found**:
   ```bash
   # Check file existence
   ls -la data/rd_sharma_complete.pdf
   
   # Verify file integrity
   file data/rd_sharma_complete.pdf
   ```

3. **Memory Issues**:
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch size in .env
   BATCH_SIZE=5
   ```

4. **OCR Quality Issues**:
   ```bash
   # Test OCR quality
   python scripts/test_ocr_quality.py data/rd_sharma_complete.pdf --pages 1 5
   ```

### Performance Optimization

1. **Increase Processing Speed**:
   - Increase `MAX_WORKERS` in configuration
   - Use SSD storage for cache
   - Optimize batch size

2. **Reduce Memory Usage**:
   - Decrease `BATCH_SIZE`
   - Clear cache regularly
   - Use smaller chunk sizes

3. **Improve Accuracy**:
   - Adjust OCR confidence threshold
   - Refine prompt templates
   - Increase validation strictness

## Security Considerations

### API Security
- Store API keys in environment variables
- Use secure file permissions
- Implement rate limiting
- Monitor API usage

### Data Security
- Secure PDF storage
- Encrypt sensitive data
- Implement access controls
- Regular security updates

### Network Security
- Use HTTPS for API calls
- Implement firewall rules
- Monitor network traffic
- Secure logging

## Backup and Recovery

### Data Backup
```bash
# Backup outputs
tar -czf backup_$(date +%Y%m%d).tar.gz outputs/

# Backup configuration
cp .env backup_env_$(date +%Y%m%d)

# Backup cache
tar -czf cache_backup_$(date +%Y%m%d).tar.gz data/cache/
```

### Recovery Procedures
```bash
# Restore from backup
tar -xzf backup_20240101.tar.gz

# Restore configuration
cp backup_env_20240101 .env

# Rebuild cache if needed
python scripts/build_index.py data/rd_sharma_complete.pdf
```

## Scaling Considerations

### Horizontal Scaling
- Multiple instances for parallel processing
- Load balancer for request distribution
- Shared storage for outputs
- Database for metadata storage

### Vertical Scaling
- Increase memory allocation
- Use faster processors
- Optimize storage I/O
- Enhance network bandwidth

## Maintenance

### Regular Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear old cache
rm -rf data/cache/*

# Rotate logs
logrotate /etc/logrotate.d/rd-sharma-extractor

# Update document index
python scripts/build_index.py data/rd_sharma_complete.pdf
```

### Performance Tuning
```bash
# Monitor system resources
htop
iotop
nethogs

# Optimize configuration
# Adjust parameters in .env based on monitoring results
```

## Support and Documentation

### Getting Help
- Check the troubleshooting section
- Review API documentation
- Examine log files
- Contact support team

### Documentation Resources
- [API Reference](API_REFERENCE.md)
- [Technical Report](TECHNICAL_REPORT.md)
- [Architecture Guide](ARCHITECTURE.md)
- [README](README.md) 