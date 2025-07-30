"""
PDF handling and processing for RD Sharma Question Extractor.

This module provides PDF loading, page extraction, and content processing
capabilities for mathematical textbook content.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..utils.logger import get_logger
from ..utils.exceptions import DocumentProcessingError
from ..config import config

logger = get_logger(__name__)


class PDFHandler:
    """Handles PDF document loading and page-level content extraction."""
    
    def __init__(self, pdf_path: Optional[str] = None):
        """
        Initialize PDF handler.
        
        Args:
            pdf_path: Path to PDF file (uses config if not provided)
        """
        self.pdf_path = Path(pdf_path or config.pdf_path)
        self.document = None
        self.page_count = 0
        self.metadata = {}
        
        logger.info(f"Initialized PDF handler for: {self.pdf_path}")
    
    def load_document(self) -> Dict[str, Any]:
        """
        Load PDF document and extract basic information.
        
        Returns:
            Document metadata and information
            
        Raises:
            DocumentProcessingError: If PDF cannot be loaded
        """
        try:
            if not self.pdf_path.exists():
                raise DocumentProcessingError(
                    f"PDF file not found: {self.pdf_path}",
                    operation="load_document"
                )
            
            # Load with PyMuPDF for metadata
            self.document = fitz.open(str(self.pdf_path))
            self.page_count = len(self.document)
            
            # Extract metadata
            self.metadata = {
                "title": self.document.metadata.get("title", "Unknown"),
                "author": self.document.metadata.get("author", "Unknown"),
                "subject": self.document.metadata.get("subject", "Mathematics"),
                "creator": self.document.metadata.get("creator", "Unknown"),
                "producer": self.document.metadata.get("producer", "Unknown"),
                "creation_date": self.document.metadata.get("creationDate", ""),
                "modification_date": self.document.metadata.get("modDate", ""),
                "page_count": self.page_count,
                "file_size": self.pdf_path.stat().st_size
            }
            
            logger.info(f"Loaded PDF: {self.metadata['title']} ({self.page_count} pages)")
            return self.metadata
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to load PDF: {str(e)}",
                operation="load_document",
                context={"pdf_path": str(self.pdf_path)}
            )
    
    def extract_page_text(self, page_number: int) -> str:
        """
        Extract text content from a specific page.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If page extraction fails
        """
        try:
            if not self.document:
                raise DocumentProcessingError(
                    "Document not loaded",
                    operation="extract_page_text",
                    page_number=page_number
                )
            
            if page_number < 0 or page_number >= self.page_count:
                raise DocumentProcessingError(
                    f"Invalid page number: {page_number}",
                    operation="extract_page_text",
                    page_number=page_number
                )
            
            page = self.document[page_number]
            text = page.get_text()
            
            logger.debug(f"Extracted {len(text)} characters from page {page_number}")
            return text
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract text from page {page_number}: {str(e)}",
                operation="extract_page_text",
                page_number=page_number
            )
    
    def extract_page_image(self, page_number: int, dpi: int = 300) -> bytes:
        """
        Extract page as image for OCR processing.
        
        Args:
            page_number: Page number (0-indexed)
            dpi: Image resolution
            
        Returns:
            Image data as bytes
            
        Raises:
            DocumentProcessingError: If image extraction fails
        """
        try:
            if not self.document:
                raise DocumentProcessingError(
                    "Document not loaded",
                    operation="extract_page_image",
                    page_number=page_number
                )
            
            page = self.document[page_number]
            mat = fitz.Matrix(dpi/72, dpi/72)  # Convert DPI to matrix
            pix = page.get_pixmap(matrix=mat)
            image_data = pix.tobytes("png")
            
            logger.debug(f"Extracted image from page {page_number} ({len(image_data)} bytes)")
            return image_data
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract image from page {page_number}: {str(e)}",
                operation="extract_page_image",
                page_number=page_number
            )
    
    def extract_page_range(self, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        Extract content from a range of pages.
        
        Args:
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (inclusive)
            
        Returns:
            List of page content dictionaries
        """
        pages = []
        
        for page_num in range(start_page, min(end_page + 1, self.page_count)):
            try:
                text = self.extract_page_text(page_num)
                pages.append({
                    "page_number": page_num,
                    "text_content": text,
                    "text_length": len(text),
                    "has_content": len(text.strip()) > 0
                })
            except DocumentProcessingError as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                pages.append({
                    "page_number": page_num,
                    "text_content": "",
                    "text_length": 0,
                    "has_content": False,
                    "error": str(e)
                })
        
        return pages
    
    def detect_text_quality(self, page_number: int) -> Dict[str, Any]:
        """
        Assess text quality for a page to determine if OCR is needed.
        
        Args:
            page_number: Page number to assess
            
        Returns:
            Quality assessment metrics
        """
        try:
            text = self.extract_page_text(page_number)
            
            # Calculate quality metrics
            total_chars = len(text)
            non_whitespace_chars = len(text.replace(" ", "").replace("\n", ""))
            lines = text.split("\n")
            avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
            
            # Detect potential OCR issues
            has_mixed_case = any(c.isupper() for c in text) and any(c.islower() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            has_math_symbols = any(symbol in text for symbol in "+-*/=()[]{}")
            
            quality_score = 0.0
            if total_chars > 100:
                quality_score += 0.3
            if has_mixed_case:
                quality_score += 0.2
            if has_numbers:
                quality_score += 0.2
            if has_math_symbols:
                quality_score += 0.2
            if avg_line_length > 20:
                quality_score += 0.1
            
            needs_ocr = quality_score < 0.5
            
            return {
                "page_number": page_number,
                "total_characters": total_chars,
                "non_whitespace_characters": non_whitespace_chars,
                "line_count": len(lines),
                "average_line_length": avg_line_length,
                "has_mixed_case": has_mixed_case,
                "has_numbers": has_numbers,
                "has_math_symbols": has_math_symbols,
                "quality_score": quality_score,
                "needs_ocr": needs_ocr
            }
            
        except Exception as e:
            logger.error(f"Failed to assess text quality for page {page_number}: {e}")
            return {
                "page_number": page_number,
                "error": str(e),
                "needs_ocr": True
            }
    
    def extract_tables(self, page_number: int) -> List[Dict[str, Any]]:
        """
        Extract tables from a specific page using pdfplumber.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            List of extracted tables
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    return []
                
                page = pdf.pages[page_number]
                tables = page.extract_tables()
                
                extracted_tables = []
                for i, table in enumerate(tables):
                    if table:
                        extracted_tables.append({
                            "table_index": i,
                            "page_number": page_number,
                            "rows": len(table),
                            "columns": len(table[0]) if table else 0,
                            "data": table
                        })
                
                return extracted_tables
                
        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_number}: {e}")
            return []
    
    def get_page_dimensions(self, page_number: int) -> Dict[str, float]:
        """
        Get page dimensions and layout information.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Page dimensions and layout info
        """
        try:
            if not self.document:
                raise DocumentProcessingError("Document not loaded")
            
            page = self.document[page_number]
            rect = page.rect
            
            return {
                "page_number": page_number,
                "width": rect.width,
                "height": rect.height,
                "rotation": page.rotation,
                "media_box": [rect.x0, rect.y0, rect.x1, rect.y1]
            }
            
        except Exception as e:
            logger.error(f"Failed to get dimensions for page {page_number}: {e}")
            return {"page_number": page_number, "error": str(e)}
    
    def close(self):
        """Close the PDF document and free resources."""
        if self.document:
            self.document.close()
            self.document = None
            logger.info("PDF document closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 