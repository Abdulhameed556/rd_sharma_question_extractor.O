# API Reference

## Overview

This document provides detailed API reference for the RD Sharma Question Extractor components.

## Core Components

### QuestionExtractor

The main class for extracting questions from RD Sharma textbook.

```python
from src.main import QuestionExtractor

extractor = QuestionExtractor()
questions = extractor.extract_questions(chapter="30", topic="30.3")
```

#### Methods

- `extract_questions(chapter: str, topic: str) -> List[Dict]`: Extract questions from specified chapter and topic
- `get_document_info() -> Dict`: Get document information and statistics
- `validate_latex(latex_content: str) -> Dict`: Validate LaTeX formatting

### Config

Configuration management using Pydantic.

```python
from src.config import Config

config = Config()
print(config.groq_api_key)
```

### GroqClient

LLM interface for Groq API.

```python
from src.llm_interface.groq_client import GroqClient

client = GroqClient(config)
response = client.extract_questions(content, chapter, topic)
```

## Document Processing

### PDFHandler

PDF loading and page extraction.

```python
from src.document_processor.pdf_handler import PDFHandler

handler = PDFHandler(config)
handler.load_document("path/to/pdf")
text = handler.extract_page_text(0)
```

### OCRProcessor

Optical Character Recognition for mathematical content.

```python
from src.document_processor.ocr_processor import OCRProcessor

processor = OCRProcessor(config)
result = processor.process_image(image)
```

## RAG Pipeline

### EmbeddingGenerator

Vector embeddings generation.

```python
from src.rag_pipeline.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(config)
embedding = generator.generate_embedding(text)
```

### VectorStore

FAISS-based vector storage.

```python
from src.rag_pipeline.vector_store import VectorStore

store = VectorStore(config)
store.add_chunks(chunks)
results = store.search(query, k=5)
```

## Question Processing

### QuestionDetector

Question identification and detection.

```python
from src.question_extractor.detector import QuestionDetector

detector = QuestionDetector(config)
questions = detector.detect_questions(content)
```

### LaTeXConverter

LaTeX formatting and conversion.

```python
from src.question_extractor.latex_converter import LaTeXConverter

converter = LaTeXConverter(config)
latex_text = converter.convert_question_to_latex(question_text)
```

## Utilities

### FileHandler

File operations and management.

```python
from src.utils.file_handler import FileHandler

handler = FileHandler()
handler.save_json(data, "output.json")
```

### Logger

Structured logging.

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
```

## Error Handling

### Custom Exceptions

- `BaseExtractorError`: Base exception class
- `DocumentProcessingError`: PDF/OCR processing errors
- `LLMInterfaceError`: LLM API errors
- `ValidationError`: Data validation errors

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Groq API key
- `PDF_PATH`: Path to RD Sharma PDF
- `OUTPUT_DIR`: Output directory
- `LOG_LEVEL`: Logging level

### Configuration File

The system uses a `.env` file for configuration:

```env
GROQ_API_KEY=your_api_key_here
PDF_PATH=data/rd_sharma_complete.pdf
OUTPUT_DIR=outputs
LOG_LEVEL=INFO
```

## CLI Interface

### Main Commands

```bash
# Extract questions
python src/main.py extract --chapter 30 --topic 30.3

# Get document info
python src/main.py info

# Validate LaTeX
python src/main.py validate --file output.tex

# Setup environment
python src/main.py setup
```

### Command Options

- `--chapter`: Chapter number
- `--topic`: Topic number
- `--output`: Output directory
- `--verbose`: Verbose logging
- `--debug`: Debug mode

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_llm_interface.py

# Run with coverage
pytest --cov=src tests/
```

### Test Fixtures

Test fixtures are available in `tests/fixtures/`:

- `sample_chapter_30_3.pdf`: Sample PDF content
- `expected_questions.json`: Expected question format
- `test_prompts.json`: Test prompt templates

## Performance

### Optimization

- Caching: OCR and embedding results are cached
- Batch processing: Multiple questions processed together
- Parallel processing: OCR and embedding generation
- Memory management: Efficient chunking and processing

### Monitoring

- Performance metrics logging
- Response time tracking
- Error rate monitoring
- Resource usage tracking

## Security

### API Key Management

- Environment variable storage
- No hardcoded credentials
- Secure API communication
- Rate limiting support

### Data Privacy

- Local processing
- No data sent to external services (except Groq API)
- Secure file handling
- Log sanitization

## Troubleshooting

### Common Issues

1. **API Key Error**: Check `GROQ_API_KEY` environment variable
2. **PDF Not Found**: Verify `PDF_PATH` configuration
3. **OCR Errors**: Check image quality and OCR settings
4. **LaTeX Errors**: Validate LaTeX syntax and formatting

### Debug Mode

Enable debug mode for detailed logging:

```bash
python src/main.py extract --debug --chapter 30 --topic 30.3
```

### Log Files

Logs are stored in `outputs/logs/`:

- `extraction.log`: Main extraction logs
- `error.log`: Error logs
- `performance.log`: Performance metrics 