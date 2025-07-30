"""
Configuration management for RD Sharma Question Extractor.

This module provides centralized configuration management with environment variable
loading, validation, and default values for all system components.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, root_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define base directory as project root (adjust if your config.py is in a different folder)
base_dir = Path(__file__).resolve().parent.parent  # For example, if config.py is in src/, project root is one level up


class Config(BaseSettings):
    """Centralized configuration for the RD Sharma Question Extractor."""

    # Groq API Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("meta-llama-4-maverick-17b", env="GROQ_MODEL")

    # Document Processing - use absolute paths via Path objects
    pdf_path: Path = Field(base_dir / "data" / "rd_sharma_complete.pdf")
    document_index_path: Path = Field(base_dir / "data" / "document_index.json")

    # Vector Database and Caching
    vector_db_path: Path = Field(base_dir / "data" / "cache" / "vector_store")
    cache_dir: Path = Field(base_dir / "data" / "cache")
    ocr_cache_dir: Path = Field(base_dir / "data" / "cache" / "ocr_cache")
    embeddings_cache_dir: Path = Field(base_dir / "data" / "cache" / "embeddings_cache")

    # LLM Parameters
    temperature: float = Field(0.1, env="TEMPERATURE")
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    top_p: float = Field(0.9, env="TOP_P")
    frequency_penalty: float = Field(0.0, env="FREQUENCY_PENALTY")
    presence_penalty: float = Field(0.0, env="PRESENCE_PENALTY")

    # RAG Pipeline Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(5, env="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")

    # OCR Configuration
    ocr_languages: List[str] = Field(["en"], env="OCR_LANGUAGES")
    ocr_gpu: bool = Field(False, env="OCR_GPU")
    ocr_confidence_threshold: float = Field(0.5, env="OCR_CONFIDENCE_THRESHOLD")

    # Output Configuration
    output_dir: Path = Field(base_dir / "outputs")
    latex_output_dir: Path = Field(base_dir / "outputs" / "latex_files")
    json_output_dir: Path = Field(base_dir / "outputs" / "extracted_questions")
    markdown_output_dir: Path = Field(base_dir / "outputs" / "markdown_files")

    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Path = Field(base_dir / "outputs" / "logs" / "extraction.log")
    error_log_file: Path = Field(base_dir / "outputs" / "logs" / "error.log")
    performance_log_file: Path = Field(base_dir / "outputs" / "logs" / "performance.log")

    # Development Configuration
    debug: bool = Field(False, env="DEBUG")
    verbose: bool = Field(False, env="VERBOSE")
    save_intermediate_results: bool = Field(True, env="SAVE_INTERMEDIATE_RESULTS")
    validate_latex: bool = Field(True, env="VALIDATE_LATEX")

    # Performance Configuration
    batch_size: int = Field(10, env="BATCH_SIZE")
    max_workers: int = Field(4, env="MAX_WORKERS")
    timeout_seconds: int = Field(30, env="TIMEOUT_SECONDS")
    retry_attempts: int = Field(3, env="RETRY_ATTEMPTS")

    @validator("ocr_languages", pre=True)
    def parse_ocr_languages(cls, v):
        """Parse OCR languages from string to list."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [item.strip() for item in v.split(",")]
        return v

    @validator("groq_api_key")
    def validate_groq_api_key(cls, v):
        """Validate Groq API key format."""
        if not v or v == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY must be set to a valid API key")
        if len(v) < 20:
            raise ValueError("GROQ_API_KEY appears to be invalid (too short)")
        return v

    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        """Validate max tokens is reasonable."""
        if v < 100 or v > 8000:
            raise ValueError("Max tokens must be between 100 and 8000")
        return v

    @root_validator(pre=True)
    def resolve_paths(cls, values):
        # Force pdf_path and document_index_path to be absolute Paths
        for path_key in ['pdf_path', 'document_index_path']:
            if path_key in values:
                values[path_key] = Path(values[path_key]).expanduser().resolve()
        return values

    def create_directories(self) -> None:
        """Create all necessary directories for the application."""
        directories = [
            self.cache_dir,
            self.ocr_cache_dir,
            self.embeddings_cache_dir,
            self.output_dir,
            self.latex_output_dir,
            self.json_output_dir,
            self.markdown_output_dir,
            self.log_file.parent,
            self.error_log_file.parent,
            self.performance_log_file.parent,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq-specific configuration."""
        return {
            "api_key": self.groq_api_key,
            "model": self.groq_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG pipeline configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_retrieval": self.top_k_retrieval,
            "similarity_threshold": self.similarity_threshold,
            "vector_db_path": self.vector_db_path,
        }

    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration."""
        return {
            "languages": self.ocr_languages,
            "gpu": self.ocr_gpu,
            "confidence_threshold": self.ocr_confidence_threshold,
        }

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return {
            "output_dir": self.output_dir,
            "latex_output_dir": self.latex_output_dir,
            "json_output_dir": self.json_output_dir,
            "markdown_output_dir": self.markdown_output_dir,
            "save_intermediate_results": self.save_intermediate_results,
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global configuration instance
config = Config()
