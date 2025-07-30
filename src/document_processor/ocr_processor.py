"""
OCR processing for mathematical content in RD Sharma Question Extractor.

This module provides OCR capabilities optimized for mathematical expressions,
symbols, and textbook content using EasyOCR and post-processing techniques.
"""

import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import DocumentProcessingError
from ..config import config

logger = get_logger(__name__)


class OCRProcessor:
    """OCR processor optimized for mathematical content recognition."""
    
    def __init__(self):
        """Initialize OCR processor with mathematical content optimization."""
        self.reader = None
        self.languages = config.ocr_languages
        self.gpu = config.ocr_gpu
        self.confidence_threshold = config.ocr_confidence_threshold
        
        # Mathematical symbol correction dictionary
        self.math_corrections = {
            # Common OCR errors for mathematical symbols
            '0': '0', 'O': '0', 'o': '0',
            '1': '1', 'l': '1', 'I': '1',
            '2': '2', 'Z': '2', 'z': '2',
            '3': '3', '8': '8', 'B': '8',
            '5': '5', 'S': '5', 's': '5',
            '6': '6', 'G': '6', 'g': '6',
            '9': '9', 'g': '9',
            '+': '+', 't': '+', 'T': '+',
            '-': '-', '_': '-',
            '=': '=', '==': '=',
            '×': '×', 'x': '×', 'X': '×',
            '÷': '÷', '/': '÷',
            '√': '√', 'V': '√',
            'π': 'π', 'pi': 'π',
            '∞': '∞', 'inf': '∞',
            '≤': '≤', '<=': '≤',
            '≥': '≥', '>=': '≥',
            '≠': '≠', '!=': '≠',
            '±': '±', '+/-': '±',
            '∑': '∑', 'sum': '∑',
            '∫': '∫', 'int': '∫',
            '∂': '∂', 'd': '∂',
            '∆': '∆', 'delta': '∆',
            'θ': 'θ', 'theta': 'θ',
            'α': 'α', 'alpha': 'α',
            'β': 'β', 'beta': 'β',
            'γ': 'γ', 'gamma': 'γ',
            'δ': 'δ', 'delta': 'δ',
            'ε': 'ε', 'epsilon': 'ε',
            'λ': 'λ', 'lambda': 'λ',
            'μ': 'μ', 'mu': 'μ',
            'σ': 'σ', 'sigma': 'σ',
            'φ': 'φ', 'phi': 'φ',
            'ψ': 'ψ', 'psi': 'ψ',
            'ω': 'ω', 'omega': 'ω'
        }
        
        # Mathematical expression patterns
        self.math_patterns = [
            r'\b\d+\s*[+\-×÷=<>≤≥≠±]\s*\d+\b',  # Basic operations
            r'\b\d+\s*[+\-×÷]\s*[a-zA-Z]\b',    # Variables
            r'\b[a-zA-Z]\s*=\s*\d+\b',          # Assignments
            r'\b\d+\s*[+\-×÷]\s*\(\s*\d+\s*[+\-×÷]\s*\d+\s*\)\b',  # Parentheses
            r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\([^)]+\)\b',      # Functions
            r'\b\d+\s*[+\-×÷]\s*\d+\s*[+\-×÷]\s*\d+\b',           # Multiple operations
        ]
        
        self._initialize_reader()
        logger.info("OCR processor initialized")
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with optimal settings for mathematical content."""
        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=str(Path(config.cache_dir) / "easyocr_models"),
                download_enabled=True,
                quantize=True  # Reduce memory usage
            )
            logger.info(f"EasyOCR reader initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise DocumentProcessingError(
                f"OCR initialization failed: {str(e)}",
                operation="initialize_reader"
            )
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Process image data and extract text with mathematical content optimization.
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            OCR results with mathematical content processing
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(cv_image)
            
            # Perform OCR
            results = self.reader.readtext(processed_image)
            
            # Post-process results
            processed_results = self._post_process_results(results)
            
            # Extract mathematical content
            math_content = self._extract_mathematical_content(processed_results)
            
            return {
                "raw_results": results,
                "processed_results": processed_results,
                "mathematical_content": math_content,
                "confidence_score": self._calculate_confidence(processed_results),
                "text_blocks": self._extract_text_blocks(processed_results)
            }
            
        except Exception as e:
            raise DocumentProcessingError(
                f"OCR processing failed: {str(e)}",
                operation="process_image"
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _post_process_results(self, results: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Post-process OCR results with mathematical symbol correction.
        
        Args:
            results: Raw OCR results from EasyOCR
            
        Returns:
            Processed results with corrections
        """
        processed = []
        
        for bbox, text, confidence in results:
            if confidence < self.confidence_threshold:
                continue
            
            # Apply mathematical symbol corrections
            corrected_text = self._correct_mathematical_symbols(text)
            
            processed.append({
                "bbox": bbox,
                "original_text": text,
                "corrected_text": corrected_text,
                "confidence": confidence,
                "is_mathematical": self._is_mathematical_expression(corrected_text)
            })
        
        return processed
    
    def _correct_mathematical_symbols(self, text: str) -> str:
        """
        Correct common OCR errors in mathematical symbols.
        
        Args:
            text: Original OCR text
            
        Returns:
            Corrected text
        """
        corrected = text
        
        # Apply symbol corrections
        for wrong, correct in self.math_corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        # Fix common mathematical expression patterns
        corrected = re.sub(r'\b(\d+)\s*[xX]\s*(\d+)\b', r'\1×\2', corrected)  # Fix multiplication
        corrected = re.sub(r'\b(\d+)\s*/\s*(\d+)\b', r'\1÷\2', corrected)     # Fix division
        corrected = re.sub(r'\b(\d+)\s*=\s*(\d+)\b', r'\1=\2', corrected)     # Fix equality
        
        return corrected
    
    def _is_mathematical_expression(self, text: str) -> bool:
        """
        Check if text contains mathematical expressions.
        
        Args:
            text: Text to check
            
        Returns:
            True if mathematical content is detected
        """
        # Check for mathematical patterns
        for pattern in self.math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for mathematical symbols
        math_symbols = ['+', '-', '×', '÷', '=', '<', '>', '≤', '≥', '≠', '±', '√', 'π', '∞', '∑', '∫', '∂', '∆']
        if any(symbol in text for symbol in math_symbols):
            return True
        
        # Check for function names
        math_functions = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'max', 'min']
        if any(func in text.lower() for func in math_functions):
            return True
        
        return False
    
    def _extract_mathematical_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and categorize mathematical content from OCR results.
        
        Args:
            results: Processed OCR results
            
        Returns:
            List of mathematical content items
        """
        math_content = []
        
        for result in results:
            if result["is_mathematical"]:
                math_content.append({
                    "text": result["corrected_text"],
                    "bbox": result["bbox"],
                    "confidence": result["confidence"],
                    "type": self._categorize_mathematical_content(result["corrected_text"])
                })
        
        return math_content
    
    def _categorize_mathematical_content(self, text: str) -> str:
        """
        Categorize mathematical content by type.
        
        Args:
            text: Mathematical text
            
        Returns:
            Content category
        """
        if re.search(r'\b\d+\s*[+\-×÷]\s*\d+\b', text):
            return "arithmetic_operation"
        elif re.search(r'\b[a-zA-Z]\s*=\s*\d+\b', text):
            return "variable_assignment"
        elif re.search(r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\([^)]+\)\b', text, re.IGNORECASE):
            return "mathematical_function"
        elif re.search(r'\b\d+\s*[+\-×÷]\s*[a-zA-Z]\b', text):
            return "algebraic_expression"
        elif any(symbol in text for symbol in ['≤', '≥', '≠', '±', '√', 'π', '∞', '∑', '∫']):
            return "advanced_mathematical"
        else:
            return "mathematical_expression"
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score for OCR results.
        
        Args:
            results: Processed OCR results
            
        Returns:
            Average confidence score
        """
        if not results:
            return 0.0
        
        total_confidence = sum(result["confidence"] for result in results)
        return total_confidence / len(results)
    
    def _extract_text_blocks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text blocks with spatial information.
        
        Args:
            results: Processed OCR results
            
        Returns:
            Text blocks with layout information
        """
        text_blocks = []
        
        for result in results:
            bbox = result["bbox"]
            
            # Calculate block properties
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            text_blocks.append({
                "text": result["corrected_text"],
                "bbox": bbox,
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "width": max(x_coords) - min(x_coords),
                "height": max(y_coords) - min(y_coords),
                "confidence": result["confidence"],
                "is_mathematical": result["is_mathematical"]
            })
        
        return text_blocks
    
    def process_page_range(self, pdf_handler, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        Process a range of pages with OCR.
        
        Args:
            pdf_handler: PDF handler instance
            start_page: Starting page number
            end_page: Ending page number
            
        Returns:
            List of OCR results for each page
        """
        results = []
        
        for page_num in range(start_page, end_page + 1):
            try:
                # Check if page needs OCR
                quality = pdf_handler.detect_text_quality(page_num)
                
                if quality.get("needs_ocr", True):
                    # Extract page image
                    image_data = pdf_handler.extract_page_image(page_num)
                    
                    # Process with OCR
                    ocr_result = self.process_image(image_data)
                    ocr_result["page_number"] = page_num
                    ocr_result["quality_assessment"] = quality
                    
                    results.append(ocr_result)
                else:
                    # Use extracted text directly
                    text = pdf_handler.extract_page_text(page_num)
                    results.append({
                        "page_number": page_num,
                        "text_content": text,
                        "ocr_used": False,
                        "quality_assessment": quality
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process page {page_num} with OCR: {e}")
                results.append({
                    "page_number": page_num,
                    "error": str(e),
                    "ocr_used": False
                })
        
        return results
    
    def save_cache(self, cache_key: str, results: List[Dict[str, Any]]):
        """Save OCR results to cache."""
        cache_file = Path(config.ocr_cache_dir) / f"{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug(f"Saved OCR cache: {cache_file}")
    
    def load_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load OCR results from cache."""
        cache_file = Path(config.ocr_cache_dir) / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                results = json.load(f)
            logger.debug(f"Loaded OCR cache: {cache_file}")
            return results
        
        return None
    
    def generate_cache_key(self, page_range: Tuple[int, int]) -> str:
        """Generate cache key for page range."""
        return hashlib.md5(f"pages_{page_range[0]}_{page_range[1]}".encode()).hexdigest() 