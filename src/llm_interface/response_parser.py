"""
Response parser for RD Sharma Question Extractor.

This module handles parsing, validation, and processing of LLM responses
from the Groq API.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger
from utils.exceptions import LLMInterfaceError
from config import config

logger = get_logger(__name__)


@dataclass
class ParsedResponse:
    """Parsed LLM response with metadata."""
    content: Any
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    processing_time: float
    token_count: Optional[int] = None
    model_used: Optional[str] = None


class ResponseParser:
    """Handles parsing and validation of LLM responses."""

    def __init__(self, config):
        """Initialize response parser."""
        self.config = config
        self.logger = get_logger(__name__)

    def parse_extraction_response(self, response_text: str, expected_format: str = "json") -> ParsedResponse:
        """
        Parse question extraction response from LLM.
        
        Args:
            response_text: Raw response text from LLM
            expected_format: Expected response format (json, text, etc.)
            
        Returns:
            ParsedResponse object with parsed content and metadata
        """
        start_time = datetime.now()
        
        try:
            # Clean the response text
            cleaned_text = self._clean_response_text(response_text)
            
            # Parse based on expected format
            if expected_format == "json":
                parsed_content = self._parse_json_response(cleaned_text)
            else:
                parsed_content = self._parse_text_response(cleaned_text)
            
            # Validate the parsed content
            validation_result = self._validate_extraction_content(parsed_content)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ParsedResponse(
                content=parsed_content,
                is_valid=validation_result['is_valid'],
                confidence=validation_result['confidence'],
                errors=validation_result['errors'],
                warnings=validation_result['warnings'],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing extraction response: {e}")
            return ParsedResponse(
                content=None,
                is_valid=False,
                confidence=0.0,
                errors=[str(e)],
                warnings=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def parse_validation_response(self, response_text: str) -> ParsedResponse:
        """
        Parse validation response from LLM.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            ParsedResponse object with validation results
        """
        start_time = datetime.now()
        
        try:
            cleaned_text = self._clean_response_text(response_text)
            parsed_content = self._parse_json_response(cleaned_text)
            
            # Validate validation response structure
            validation_result = self._validate_validation_response(parsed_content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ParsedResponse(
                content=parsed_content,
                is_valid=validation_result['is_valid'],
                confidence=validation_result.get('score', 0.0) / 100.0,
                errors=validation_result.get('errors', []),
                warnings=validation_result.get('warnings', []),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing validation response: {e}")
            return ParsedResponse(
                content=None,
                is_valid=False,
                confidence=0.0,
                errors=[str(e)],
                warnings=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def parse_latex_response(self, response_text: str) -> ParsedResponse:
        """
        Parse LaTeX formatting response from LLM.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            ParsedResponse object with LaTeX formatted content
        """
        start_time = datetime.now()
        
        try:
            cleaned_text = self._clean_response_text(response_text)
            
            # Validate LaTeX formatting
            validation_result = self._validate_latex_formatting(cleaned_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ParsedResponse(
                content=cleaned_text,
                is_valid=validation_result['is_valid'],
                confidence=validation_result['confidence'],
                errors=validation_result['errors'],
                warnings=validation_result['warnings'],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LaTeX response: {e}")
            return ParsedResponse(
                content=None,
                is_valid=False,
                confidence=0.0,
                errors=[str(e)],
                warnings=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def _clean_response_text(self, response_text: str) -> str:
        """Clean and normalize response text."""
        if not response_text:
            raise LLMInterfaceError("Empty response text")
        
        # Remove leading/trailing whitespace
        cleaned = response_text.strip()
        
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'```latex\s*', '', cleaned)
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^Here is the (JSON|LaTeX|formatted) (response|output):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^The (JSON|LaTeX|formatted) (response|output) is:\s*', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _parse_json_response(self, text: str) -> Any:
        """Parse JSON response with error handling."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from malformed response
            extracted_json = self._extract_json_from_text(text)
            if extracted_json:
                return json.loads(extracted_json)
            else:
                raise LLMInterfaceError(f"Failed to parse JSON response: {e}")

    def _parse_text_response(self, text: str) -> str:
        """Parse plain text response."""
        return text

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from malformed text response."""
        # Look for JSON array or object patterns
        json_patterns = [
            r'\[.*\]',  # JSON array
            r'\{.*\}',  # JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None

    def _validate_extraction_content(self, content: Any) -> Dict[str, Any]:
        """Validate extracted question content."""
        errors = []
        warnings = []
        confidence = 1.0
        
        if not isinstance(content, list):
            errors.append("Content is not a list")
            confidence *= 0.5
        
        if not content:
            warnings.append("No questions extracted")
            confidence *= 0.8
        
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                errors.append(f"Item {i} is not a dictionary")
                confidence *= 0.9
                continue
            
            # Check required fields
            required_fields = ['question_number', 'question_text', 'source']
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i} missing required field: {field}")
                    confidence *= 0.9
            
            # Validate question text
            if 'question_text' in item:
                question_text = item['question_text']
                if not isinstance(question_text, str):
                    errors.append(f"Item {i} question_text is not a string")
                    confidence *= 0.9
                elif len(question_text.strip()) < 10:
                    warnings.append(f"Item {i} question_text is very short")
                    confidence *= 0.95
                
                # Check for LaTeX formatting
                if not self._has_latex_formatting(question_text):
                    warnings.append(f"Item {i} may lack proper LaTeX formatting")
                    confidence *= 0.9
        
        return {
            'is_valid': len(errors) == 0,
            'confidence': max(0.0, confidence),
            'errors': errors,
            'warnings': warnings
        }

    def _validate_validation_response(self, content: Any) -> Dict[str, Any]:
        """Validate validation response structure."""
        errors = []
        warnings = []
        
        if not isinstance(content, dict):
            errors.append("Validation response is not a dictionary")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        required_fields = ['is_valid', 'score']
        for field in required_fields:
            if field not in content:
                errors.append(f"Missing required field: {field}")
        
        if 'score' in content:
            score = content['score']
            if not isinstance(score, (int, float)) or score < 0 or score > 100:
                errors.append("Score must be a number between 0 and 100")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _validate_latex_formatting(self, text: str) -> Dict[str, Any]:
        """Validate LaTeX formatting quality."""
        errors = []
        warnings = []
        confidence = 1.0
        
        # Check for basic LaTeX patterns
        if not re.search(r'\$.*\$', text):
            warnings.append("No LaTeX math mode detected")
            confidence *= 0.8
        
        # Check for unmatched dollar signs
        dollar_count = text.count('$')
        if dollar_count % 2 != 0:
            errors.append("Unmatched dollar signs in LaTeX")
            confidence *= 0.7
        
        # Check for common LaTeX commands
        latex_commands = [r'\\text\{', r'\\frac\{', r'P\(', r'\\cap', r'\\cup']
        for command in latex_commands:
            if re.search(command, text):
                confidence *= 1.1  # Bonus for good LaTeX usage
        
        # Check for numbers not in LaTeX
        numbers_not_in_latex = re.findall(r'(?<!\$)\b\d+\b(?!\$)', text)
        if numbers_not_in_latex:
            warnings.append(f"Numbers not in LaTeX format: {numbers_not_in_latex}")
            confidence *= 0.9
        
        return {
            'is_valid': len(errors) == 0,
            'confidence': min(1.0, confidence),
            'errors': errors,
            'warnings': warnings
        }

    def _has_latex_formatting(self, text: str) -> bool:
        """Check if text has LaTeX formatting."""
        # Check for basic LaTeX patterns
        latex_patterns = [
            r'\$.*\$',  # Math mode
            r'\\text\{',  # Text command
            r'\\frac\{',  # Fraction
            r'P\(',  # Probability
            r'\\cap',  # Intersection
            r'\\cup',  # Union
        ]
        
        return any(re.search(pattern, text) for pattern in latex_patterns)

    def extract_questions_from_response(self, parsed_response: ParsedResponse) -> List[Dict[str, str]]:
        """Extract questions from parsed response."""
        if not parsed_response.is_valid or not parsed_response.content:
            return []
        
        questions = []
        for item in parsed_response.content:
            if isinstance(item, dict) and 'question_text' in item:
                questions.append({
                    'question_number': item.get('question_number', 'Unknown'),
                    'question_text': item['question_text'],
                    'source': item.get('source', 'Unknown')
                })
        
        return questions

    def get_response_statistics(self, parsed_response: ParsedResponse) -> Dict[str, Any]:
        """Get statistics about parsed response."""
        stats = {
            'is_valid': parsed_response.is_valid,
            'confidence': parsed_response.confidence,
            'processing_time': parsed_response.processing_time,
            'error_count': len(parsed_response.errors),
            'warning_count': len(parsed_response.warnings)
        }
        
        if parsed_response.content:
            if isinstance(parsed_response.content, list):
                stats['item_count'] = len(parsed_response.content)
            elif isinstance(parsed_response.content, str):
                stats['text_length'] = len(parsed_response.content)
        
        return stats

    def format_response_for_logging(self, parsed_response: ParsedResponse) -> str:
        """Format response for logging purposes."""
        lines = [
            f"Response Status: {'VALID' if parsed_response.is_valid else 'INVALID'}",
            f"Confidence: {parsed_response.confidence:.2f}",
            f"Processing Time: {parsed_response.processing_time:.3f}s",
            f"Errors: {len(parsed_response.errors)}",
            f"Warnings: {len(parsed_response.warnings)}"
        ]
        
        if parsed_response.errors:
            lines.append("Error Details:")
            for error in parsed_response.errors:
                lines.append(f"  - {error}")
        
        if parsed_response.warnings:
            lines.append("Warning Details:")
            for warning in parsed_response.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines) 