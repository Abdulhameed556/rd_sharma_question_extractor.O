"""
Custom exceptions for RD Sharma Question Extractor.

This module defines a hierarchical exception structure for different types
of errors that can occur during the extraction process.
"""

from typing import Optional, Dict, Any


class BaseExtractorError(Exception):
    """Base exception for all RD Sharma Question Extractor errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class DocumentProcessingError(BaseExtractorError):
    """Raised when there are issues with PDF loading, OCR, or document parsing."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        page_number: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        if page_number is not None:
            context["page_number"] = page_number
        
        super().__init__(
            message=message,
            error_code="DOC_PROC_001",
            context=context
        )


class RAGPipelineError(BaseExtractorError):
    """Raised when there are issues with the RAG pipeline (embeddings, vector store, retrieval)."""
    
    def __init__(
        self, 
        message: str, 
        component: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if component:
            context["component"] = component
        if query:
            context["query"] = query
        
        super().__init__(
            message=message,
            error_code="RAG_001",
            context=context
        )


class LLMInterfaceError(BaseExtractorError):
    """Raised when there are issues with LLM API calls, responses, or model interactions."""
    
    def __init__(
        self, 
        message: str, 
        model: Optional[str] = None,
        api_response: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if model:
            context["model"] = model
        if api_response:
            context["api_response"] = api_response
        
        super().__init__(
            message=message,
            error_code="LLM_001",
            context=context
        )


class ValidationError(BaseExtractorError):
    """Raised when output validation fails (LaTeX syntax, question format, etc.)."""
    
    def __init__(
        self, 
        message: str, 
        validation_type: Optional[str] = None,
        invalid_content: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if validation_type:
            context["validation_type"] = validation_type
        if invalid_content:
            context["invalid_content"] = invalid_content
        
        super().__init__(
            message=message,
            error_code="VAL_001",
            context=context
        )


class ConfigurationError(BaseExtractorError):
    """Raised when there are configuration issues."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIG_001",
            context=context
        )


class FileOperationError(BaseExtractorError):
    """Raised when there are file operation issues."""
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if file_path:
            context["file_path"] = file_path
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="FILE_001",
            context=context
        )


# Error code mapping for easy reference
ERROR_CODES = {
    "DOC_PROC_001": "Document processing error",
    "RAG_001": "RAG pipeline error", 
    "LLM_001": "LLM interface error",
    "VAL_001": "Validation error",
    "CONFIG_001": "Configuration error",
    "FILE_001": "File operation error"
}


def get_error_description(error_code: str) -> str:
    """Get human-readable description for error codes."""
    return ERROR_CODES.get(error_code, "Unknown error") 