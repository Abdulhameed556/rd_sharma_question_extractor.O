# Troubleshooting Guide

## Common Issues and Solutions

This guide provides solutions to common problems encountered when using the RD Sharma Question Extractor.

## Quick Diagnosis

### Check System Status
```bash
# Verify installation
python src/main.py info

# Check configuration
python src/main.py config

# Test API connectivity
python src/main.py test-api
```

## Installation Issues

### Python Version Problems

**Problem**: `SyntaxError` or import errors
```bash
# Check Python version
python --version

# Solution: Use Python 3.8+
python3.8 -m venv venv
source venv/bin/activate
```

### Dependency Installation Failures

**Problem**: `pip install` fails
```bash
# Update pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Alternative: Use conda
conda env create -f environment.yml
```

### Missing System Dependencies

**Problem**: OCR or PDF processing fails
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download and install Tesseract from GitHub releases
```

## Configuration Issues

### API Key Problems

**Problem**: `GROQ_API_KEY` not found
```bash
# Check environment variable
echo $GROQ_API_KEY

# Verify .env file exists
ls -la .env

# Create .env file
cp env.example .env
# Edit .env and add your API key
```

**Problem**: Invalid API key
```bash
# Test API key
python src/main.py test-api

# Solution: Generate new key from Groq console
# https://console.groq.com/
```

### File Path Issues

**Problem**: PDF file not found
```bash
# Check file existence
ls -la data/rd_sharma_complete.pdf

# Verify file permissions
chmod 644 data/rd_sharma_complete.pdf

# Check file integrity
file data/rd_sharma_complete.pdf
```

## Processing Issues

### OCR Quality Problems

**Problem**: Poor text recognition
```bash
# Test OCR quality
python scripts/test_ocr_quality.py data/rd_sharma_complete.pdf

# Solutions:
# 1. Improve PDF quality
# 2. Adjust OCR confidence threshold
# 3. Use different OCR engine
```

**Problem**: Mathematical symbols not recognized
```bash
# Check OCR preprocessing
python scripts/test_ocr_quality.py data/rd_sharma_complete.pdf --debug

# Solutions:
# 1. Increase image resolution
# 2. Improve contrast
# 3. Use specialized math OCR
```

### LLM Processing Issues

**Problem**: API rate limiting
```bash
# Check rate limits
python src/main.py info

# Solutions:
# 1. Reduce batch size
# 2. Add delays between requests
# 3. Use fallback models
```

**Problem**: Invalid JSON responses
```bash
# Check response parsing
python src/main.py extract --chapter 30 --topic 30.3 --debug

# Solutions:
# 1. Refine prompt templates
# 2. Add response validation
# 3. Implement retry logic
```

### Memory Issues

**Problem**: Out of memory errors
```bash
# Check memory usage
htop
free -h

# Solutions:
# 1. Reduce batch size in .env
# 2. Clear cache: rm -rf data/cache/*
# 3. Process smaller chunks
# 4. Increase system memory
```

## Output Issues

### LaTeX Formatting Problems

**Problem**: Incorrect LaTeX syntax
```bash
# Validate LaTeX output
python src/main.py validate --file outputs/latex_files/chapter_30_3.tex

# Solutions:
# 1. Check LaTeX converter
# 2. Refine prompt templates
# 3. Add syntax validation
```

**Problem**: Missing mathematical expressions
```bash
# Check question extraction
python src/main.py extract --chapter 30 --topic 30.3 --validate

# Solutions:
# 1. Improve question detection
# 2. Enhance LaTeX conversion
# 3. Add post-processing
```

### File Generation Issues

**Problem**: Output files not created
```bash
# Check output directory
ls -la outputs/

# Check permissions
chmod 755 outputs/
chmod 755 outputs/extracted_questions/
chmod 755 outputs/latex_files/

# Check disk space
df -h
```

## Performance Issues

### Slow Processing

**Problem**: Extraction takes too long
```bash
# Monitor performance
tail -f outputs/logs/performance.log

# Solutions:
# 1. Increase MAX_WORKERS in .env
# 2. Use SSD storage
# 3. Optimize batch size
# 4. Enable caching
```

### High Memory Usage

**Problem**: Excessive memory consumption
```bash
# Monitor memory
htop
ps aux | grep python

# Solutions:
# 1. Reduce BATCH_SIZE
# 2. Clear cache regularly
# 3. Use smaller chunk sizes
# 4. Implement garbage collection
```

## Network Issues

### API Connectivity Problems

**Problem**: Cannot connect to Groq API
```bash
# Test connectivity
curl -I https://api.groq.com

# Check firewall
sudo ufw status

# Solutions:
# 1. Check internet connection
# 2. Configure proxy if needed
# 3. Update DNS settings
```

### Timeout Errors

**Problem**: API requests timeout
```bash
# Check timeout settings
grep TIMEOUT .env

# Solutions:
# 1. Increase TIMEOUT_SECONDS
# 2. Check network stability
# 3. Use retry mechanism
```

## Log Analysis

### Understanding Log Messages

```bash
# View recent logs
tail -n 50 outputs/logs/extraction.log

# Search for errors
grep ERROR outputs/logs/extraction.log

# Search for warnings
grep WARNING outputs/logs/extraction.log

# Monitor real-time
tail -f outputs/logs/extraction.log
```

### Common Log Patterns

**High OCR Confidence**: `OCR confidence: 95.2%`
- Good: High confidence indicates good text recognition
- Action: None needed

**Low OCR Confidence**: `OCR confidence: 45.3%`
- Problem: Poor text recognition
- Action: Improve PDF quality or adjust preprocessing

**API Success**: `LLM response received successfully`
- Good: API call completed
- Action: None needed

**API Error**: `LLM request failed: Rate limit exceeded`
- Problem: API rate limiting
- Action: Reduce request frequency or use fallback

## Debug Mode

### Enable Debug Logging

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run with debug output
python src/main.py extract --chapter 30 --topic 30.3 --debug

# View debug logs
tail -f outputs/logs/extraction.log
```

### Debug Information

Debug mode provides:
- Detailed processing steps
- Intermediate results
- Performance metrics
- Error stack traces
- Configuration values

## Recovery Procedures

### Reset System State

```bash
# Clear all cache
rm -rf data/cache/*

# Clear outputs
rm -rf outputs/extracted_questions/*
rm -rf outputs/latex_files/*

# Reset logs
> outputs/logs/extraction.log
> outputs/logs/error.log
> outputs/logs/performance.log
```

### Rebuild Index

```bash
# Rebuild document index
python scripts/build_index.py data/rd_sharma_complete.pdf

# Verify index
python src/main.py info
```

### Restore from Backup

```bash
# Restore configuration
cp backup_env_20240101 .env

# Restore outputs
tar -xzf backup_20240101.tar.gz

# Restore cache
tar -xzf cache_backup_20240101.tar.gz
```

## Getting Help

### Self-Service Resources

1. **Check Documentation**:
   - [README](README.md)
   - [API Reference](API_REFERENCE.md)
   - [Architecture Guide](ARCHITECTURE.md)

2. **Review Logs**:
   - Check all log files in `outputs/logs/`
   - Look for error patterns
   - Monitor performance metrics

3. **Run Diagnostics**:
   ```bash
   python src/main.py info
   python src/main.py config
   python src/main.py test-api
   ```

### Contact Support

If self-service solutions don't resolve the issue:

1. **Collect Information**:
   - System information
   - Error logs
   - Configuration files
   - Steps to reproduce

2. **Submit Issue**:
   - Create detailed bug report
   - Include all relevant information
   - Provide reproduction steps

3. **Follow Up**:
   - Monitor issue status
   - Provide additional information if requested
   - Test proposed solutions

## Prevention

### Best Practices

1. **Regular Maintenance**:
   - Update dependencies monthly
   - Clear cache weekly
   - Monitor disk space
   - Check log files

2. **Configuration Management**:
   - Use version control for configuration
   - Document changes
   - Test changes in development
   - Backup configurations

3. **Monitoring**:
   - Set up performance monitoring
   - Monitor API usage
   - Track error rates
   - Monitor resource usage

4. **Testing**:
   - Run tests regularly
   - Test with different inputs
   - Validate outputs
   - Performance testing 