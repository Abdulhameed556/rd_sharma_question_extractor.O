# System Architecture

## Overview

The RD Sharma Question Extractor implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed to extract mathematical questions from textbook content and format them in precise LaTeX.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │ Processing Layer│    │  Output Layer   │
│                 │    │                 │    │                 │
│ • Chapter/Topic │───▶│ • Document Proc │───▶│ • JSON Output   │
│ • PDF Document  │    │ • RAG Pipeline  │    │ • LaTeX Output  │
│ • Configuration │    │ • LLM Interface │    │ • Markdown      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Validation &   │
                       │   Quality       │
                       │   Assurance     │
                       └─────────────────┘
```

## Core Components

### 1. Document Processor (`src/document_processor/`)

**Purpose**: Handles PDF loading, OCR processing, and content analysis.

**Components**:
- **PDFHandler**: PDF loading and page extraction using PyMuPDF
- **OCRProcessor**: Mathematical content recognition using EasyOCR
- **DocumentIndexer**: Chapter/topic mapping and structure analysis
- **ContentParser**: Text structure analysis and classification

**Data Flow**:
```
PDF Input → Page Extraction → OCR Processing → Content Analysis → Structured Output
```

### 2. RAG Pipeline (`src/rag_pipeline/`)

**Purpose**: Implements the retrieval-augmented generation system.

**Components**:
- **EmbeddingGenerator**: Vector embeddings using sentence-transformers
- **ContentChunker**: Intelligent content chunking with overlap
- **VectorStore**: FAISS-based vector storage and retrieval
- **RAGRetriever**: Dynamic boundary detection and content retrieval

**Data Flow**:
```
Content → Chunking → Embedding → Vector Storage → Retrieval → Context
```

### 3. LLM Interface (`src/llm_interface/`)

**Purpose**: Manages interactions with Groq's Meta-Llama-4-Maverick-17B model.

**Components**:
- **GroqClient**: API integration with retry logic and rate limiting
- **PromptTemplates**: Optimized prompt templates for different tasks
- **ResponseParser**: JSON response parsing and validation
- **FallbackHandler**: Backup model management and failover

**Data Flow**:
```
Context → Prompt Building → LLM Request → Response Parsing → Structured Output
```

### 4. Question Extractor (`src/question_extractor/`)

**Purpose**: Detects and processes mathematical questions.

**Components**:
- **QuestionDetector**: Multi-strategy question identification
- **LaTeXConverter**: Mathematical expression formatting
- **QuestionValidator**: Quality assurance and validation

**Data Flow**:
```
Content → Question Detection → LaTeX Conversion → Validation → Final Output
```

### 5. Utilities (`src/utils/`)

**Purpose**: Shared utilities and infrastructure.

**Components**:
- **Logger**: Structured logging with performance metrics
- **Exceptions**: Custom exception hierarchy
- **FileHandler**: Centralized file operations
- **LaTeXRenderer**: LaTeX validation and rendering

## Data Flow Architecture

### 1. Initialization Phase
```
Configuration Loading → Environment Setup → Component Initialization
```

### 2. Document Processing Phase
```
PDF Loading → Page Extraction → OCR Processing → Content Analysis
```

### 3. RAG Processing Phase
```
Content Chunking → Embedding Generation → Vector Storage → Retrieval
```

### 4. LLM Processing Phase
```
Context Assembly → Prompt Building → LLM Request → Response Processing
```

### 5. Output Generation Phase
```
Question Extraction → LaTeX Formatting → Validation → File Generation
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pydantic**: Configuration management and validation
- **Typer**: CLI interface framework
- **Rich**: Terminal output formatting

### Document Processing
- **PyMuPDF**: PDF loading and manipulation
- **EasyOCR**: Optical character recognition
- **Pillow**: Image processing
- **NumPy**: Numerical operations

### RAG Pipeline
- **Sentence-Transformers**: Vector embeddings
- **FAISS**: Vector similarity search
- **LangChain**: RAG framework integration
- **Pandas**: Data manipulation

### LLM Integration
- **Groq API**: Meta-Llama-4-Maverick-17B access
- **Requests**: HTTP client for API calls
- **JSON**: Response parsing and validation

### Testing & Quality
- **Pytest**: Testing framework
- **Coverage**: Code coverage analysis
- **Black**: Code formatting
- **Flake8**: Code linting

## Performance Architecture

### Caching Strategy
- **OCR Cache**: Stores processed OCR results
- **Embedding Cache**: Caches vector embeddings
- **Index Cache**: Document structure metadata
- **Response Cache**: LLM response caching

### Optimization Techniques
- **Batch Processing**: Multiple questions processed together
- **Parallel Processing**: OCR and embedding generation
- **Memory Management**: Efficient chunking and processing
- **Lazy Loading**: Components loaded on demand

### Scalability Considerations
- **Modular Design**: Independent component scaling
- **Resource Pooling**: Shared resource management
- **Error Recovery**: Graceful failure handling
- **Monitoring**: Performance metrics tracking

## Security Architecture

### API Security
- **Environment Variables**: Secure credential storage
- **Rate Limiting**: API call throttling
- **Error Handling**: Secure error messages
- **Input Validation**: Comprehensive input sanitization

### Data Privacy
- **Local Processing**: Minimal external data transmission
- **Secure Storage**: Encrypted cache storage
- **Log Sanitization**: Sensitive data removal
- **Access Control**: File permission management

## Deployment Architecture

### Development Environment
- **Virtual Environment**: Isolated Python environment
- **Dependency Management**: Requirements.txt and environment.yml
- **Development Tools**: VSCode configuration
- **Testing Framework**: Comprehensive test suite

### Production Environment
- **Containerization**: Docker support (future)
- **Configuration Management**: Environment-based configuration
- **Logging**: Structured logging with rotation
- **Monitoring**: Performance and error monitoring

## Error Handling Architecture

### Exception Hierarchy
```
BaseExtractorError
├── DocumentProcessingError
├── RAGPipelineError
├── LLMInterfaceError
├── ValidationError
└── ConfigurationError
```

### Recovery Mechanisms
- **Retry Logic**: Exponential backoff for API calls
- **Fallback Models**: Backup LLM models
- **Graceful Degradation**: Partial functionality on errors
- **Error Reporting**: Detailed error logging and reporting

## Future Enhancements

### Planned Improvements
- **Web Interface**: Flask/FastAPI web application
- **Database Integration**: PostgreSQL for metadata storage
- **Real-time Processing**: Streaming question extraction
- **Multi-language Support**: Hindi/English content processing

### Scalability Roadmap
- **Microservices**: Component-based deployment
- **Cloud Integration**: AWS/Azure deployment
- **Load Balancing**: Multiple instance support
- **Auto-scaling**: Dynamic resource allocation 