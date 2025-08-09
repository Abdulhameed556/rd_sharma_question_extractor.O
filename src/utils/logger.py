"""
Logging configuration for RD Sharma Question Extractor.

This module provides structured logging with JSON formatting, performance metrics
tracking, and error categorization for the extraction pipeline.
"""

import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from colorama import Fore, Style, init

from .exceptions import BaseExtractorError
from ..config import config

# Initialize colorama for colored output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Performance tracking logger for timing operations."""
    
    def __init__(self, logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timers[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and log the duration."""
        if operation not in self.timers:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        duration = time.time() - self.timers[operation]
        del self.timers[operation]
        
        extra_fields = {"duration_seconds": duration, "operation": operation}
        if extra_info:
            extra_fields.update(extra_info)
        
        self.logger.info(
            f"Completed {operation} in {duration:.2f}s",
            extra={"extra_fields": extra_fields}
        )
        
        return duration
    
    def log_operation(self, operation: str, func, *args, **kwargs):
        """Decorator-style operation logging."""
        self.start_timer(operation)
        try:
            result = func(*args, **kwargs)
            self.end_timer(operation, {"status": "success"})
            return result
        except Exception as e:
            self.end_timer(operation, {"status": "error", "error": str(e)})
            raise


def setup_logger(
    name: str = "rd_sharma_extractor",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    error_log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to main log file
        error_log_file: Path to error log file
    
    Returns:
        Configured logger instance
    """
    # Use config values if not provided
    log_level = log_level or config.log_level
    log_file = log_file or config.log_file
    error_log_file = error_log_file or config.error_log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler
    if error_log_file:
        error_path = Path(error_log_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = JSONFormatter()
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = "rd_sharma_extractor") -> logging.Logger:
    """Get a logger instance, creating it if it doesn't exist."""
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        setup_logger(name)
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an exception with context information."""
    if isinstance(exception, BaseExtractorError):
        error_info = exception.to_dict()
        if context:
            error_info["context"].update(context)
        
        logger.error(
            f"Extractor error: {exception.message}",
            extra={"extra_fields": error_info}
        )
    else:
        logger.error(
            f"Unexpected error: {str(exception)}",
            extra={"extra_fields": {"error_type": type(exception).__name__, "context": context or {}}},
            exc_info=True
        )


def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log performance metrics."""
    logger.info(
        "Performance metrics",
        extra={"extra_fields": {"metrics": metrics, "type": "performance"}}
    )


def log_extraction_start(logger: logging.Logger, chapter: int, topic: str):
    """Log the start of an extraction process."""
    logger.info(
        f"Starting extraction for Chapter {chapter}, Topic {topic}",
        extra={"extra_fields": {
            "operation": "extraction_start",
            "chapter": chapter,
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat()
        }}
    )


def log_extraction_complete(logger: logging.Logger, chapter: int, topic: str, question_count: int, duration: float):
    """Log the completion of an extraction process."""
    logger.info(
        f"Extraction completed: {question_count} questions extracted in {duration:.2f}s",
        extra={"extra_fields": {
            "operation": "extraction_complete",
            "chapter": chapter,
            "topic": topic,
            "question_count": question_count,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat()
        }}
    )


# Global logger instance
logger = get_logger()
performance_logger = PerformanceLogger(logger) 