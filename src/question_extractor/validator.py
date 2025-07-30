"""
Question validator for RD Sharma Question Extractor.

This module provides comprehensive validation capabilities for
extracted questions and LaTeX formatting.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError
from ..config import config

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Represents a validation result."""
    is_valid: bool
    score: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]


class QuestionValidator:
    """Validates extracted questions and LaTeX formatting."""
    
    def __init__(self):
        """Initialize question validator with validation rules."""
        # Question quality patterns
        self.question_patterns = {
            "has_question_marker": [
                r'\b(find|calculate|determine|evaluate|compute|solve|prove|show|verify|check)\b',
                r'\b(what|how|which|where|when|why|if|given)\b',
                r'\?',  # Question mark
            ],
            "has_mathematical_content": [
                r'\d+',  # Numbers
                r'[+\-×÷=<>≤≥≠±]',  # Mathematical operators
                r'\b(sin|cos|tan|log|ln|exp|sqrt)\b',  # Functions
                r'\bP\([^)]*\)\b',  # Probability
                r'[a-zA-Z]\s*=\s*\d+',  # Variable assignments
            ],
            "has_proper_structure": [
                r'^[A-Z]',  # Starts with capital letter
                r'[.!?]$',  # Ends with punctuation
                r'\b\d+\.\s*[A-Z]',  # Numbered questions
            ]
        }
        
        # LaTeX validation patterns
        self.latex_patterns = {
            "valid_delimiters": r'\$[^$]*\$',
            "valid_probability": r'\$P\(\\text\{[^}]*\}\)\$',
            "valid_numbers": r'\$\d+\$',
            "valid_operators": r'\\[a-z]+',
        }
        
        # Quality thresholds
        self.thresholds = {
            "min_question_length": 20,
            "max_question_length": 500,
            "min_math_content": 0.1,
            "min_question_score": 0.6,
            "min_latex_score": 0.8,
        }
        
        logger.info("Question validator initialized")
    
    def validate_question(self, question: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single question.
        
        Args:
            question: Question dictionary
            
        Returns:
            Validation result
        """
        try:
            question_text = question.get("question_text", "")
            question_number = question.get("question_number", "")
            source = question.get("source", "")
            
            # Initialize validation
            errors = []
            warnings = []
            suggestions = []
            score = 0.0
            
            # Check basic structure
            structure_validation = self._validate_structure(question_text, question_number, source)
            errors.extend(structure_validation["errors"])
            warnings.extend(structure_validation["warnings"])
            suggestions.extend(structure_validation["suggestions"])
            score += structure_validation["score"] * 0.3
            
            # Check mathematical content
            math_validation = self._validate_mathematical_content(question_text)
            errors.extend(math_validation["errors"])
            warnings.extend(math_validation["warnings"])
            suggestions.extend(math_validation["suggestions"])
            score += math_validation["score"] * 0.4
            
            # Check LaTeX formatting
            latex_validation = self._validate_latex_formatting(question_text)
            errors.extend(latex_validation["errors"])
            warnings.extend(latex_validation["warnings"])
            suggestions.extend(latex_validation["suggestions"])
            score += latex_validation["score"] * 0.3
            
            # Determine overall validity
            is_valid = score >= self.thresholds["min_question_score"] and len(errors) == 0
            
            # Create metadata
            metadata = {
                "question_length": len(question_text),
                "has_question_number": bool(question_number),
                "has_source": bool(source),
                "structure_score": structure_validation["score"],
                "math_score": math_validation["score"],
                "latex_score": latex_validation["score"],
            }
            
            return ValidationResult(
                is_valid=is_valid,
                score=score,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=[],
                metadata={}
            )
    
    def _validate_structure(self, question_text: str, question_number: str, source: str) -> Dict[str, Any]:
        """Validate question structure."""
        validation = {
            "score": 0.0,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        score = 0.0
        
        # Check length
        if len(question_text) < self.thresholds["min_question_length"]:
            validation["errors"].append(f"Question too short ({len(question_text)} chars)")
        elif len(question_text) > self.thresholds["max_question_length"]:
            validation["warnings"].append(f"Question very long ({len(question_text)} chars)")
        else:
            score += 0.2
        
        # Check for question markers
        has_question_marker = False
        for pattern in self.question_patterns["has_question_marker"]:
            if re.search(pattern, question_text, re.IGNORECASE):
                has_question_marker = True
                break
        
        if has_question_marker:
            score += 0.3
        else:
            validation["warnings"].append("No clear question indicator found")
        
        # Check for proper structure
        has_proper_structure = True
        for pattern in self.question_patterns["has_proper_structure"]:
            if not re.search(pattern, question_text):
                has_proper_structure = False
                break
        
        if has_proper_structure:
            score += 0.3
        else:
            validation["suggestions"].append("Consider improving question structure")
        
        # Check question number
        if question_number:
            score += 0.1
        else:
            validation["warnings"].append("No question number provided")
        
        # Check source
        if source:
            score += 0.1
        else:
            validation["warnings"].append("No source information")
        
        validation["score"] = min(score, 1.0)
        return validation
    
    def _validate_mathematical_content(self, question_text: str) -> Dict[str, Any]:
        """Validate mathematical content."""
        validation = {
            "score": 0.0,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        score = 0.0
        
        # Count mathematical elements
        math_elements = 0
        total_words = len(question_text.split())
        
        for pattern in self.question_patterns["has_mathematical_content"]:
            matches = re.findall(pattern, question_text, re.IGNORECASE)
            math_elements += len(matches)
        
        # Calculate math density
        math_density = math_elements / max(total_words, 1)
        
        if math_density >= self.thresholds["min_math_content"]:
            score += 0.5
        else:
            validation["errors"].append("Insufficient mathematical content")
        
        # Check for specific mathematical patterns
        has_numbers = bool(re.search(r'\d+', question_text))
        has_operators = bool(re.search(r'[+\-×÷=<>≤≥≠±]', question_text))
        has_probability = bool(re.search(r'\bP\([^)]*\)\b', question_text))
        has_functions = bool(re.search(r'\b(sin|cos|tan|log|ln|exp|sqrt)\b', question_text, re.IGNORECASE))
        
        if has_numbers:
            score += 0.2
        if has_operators:
            score += 0.2
        if has_probability:
            score += 0.1
        if has_functions:
            score += 0.1
        
        validation["score"] = min(score, 1.0)
        return validation
    
    def _validate_latex_formatting(self, question_text: str) -> Dict[str, Any]:
        """Validate LaTeX formatting."""
        validation = {
            "score": 0.0,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        score = 0.0
        
        # Check for LaTeX delimiters
        dollar_count = question_text.count('$')
        if dollar_count % 2 == 0 and dollar_count > 0:
            score += 0.3
        elif dollar_count % 2 != 0:
            validation["errors"].append("Unmatched LaTeX delimiters")
        
        # Check for valid LaTeX patterns
        valid_patterns = 0
        total_patterns = 0
        
        for pattern_name, pattern in self.latex_patterns.items():
            matches = re.findall(pattern, question_text)
            if matches:
                valid_patterns += 1
            total_patterns += 1
        
        if total_patterns > 0:
            pattern_score = valid_patterns / total_patterns
            score += pattern_score * 0.4
        
        # Check for common LaTeX issues
        if '\\text{' in question_text:
            score += 0.2
        else:
            validation["suggestions"].append("Consider using \\text{} for descriptive text in math")
        
        # Check for mathematical symbols
        math_symbols = ['\\times', '\\div', '\\leq', '\\geq', '\\neq', '\\pm']
        for symbol in math_symbols:
            if symbol in question_text:
                score += 0.1
                break
        
        validation["score"] = min(score, 1.0)
        return validation
    
    def validate_question_batch(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Batch validation results
        """
        batch_results = {
            "total_questions": len(questions),
            "valid_questions": 0,
            "invalid_questions": 0,
            "average_score": 0.0,
            "question_results": [],
            "overall_errors": [],
            "overall_warnings": [],
            "overall_suggestions": []
        }
        
        if not questions:
            return batch_results
        
        total_score = 0.0
        all_errors = set()
        all_warnings = set()
        all_suggestions = set()
        
        for question in questions:
            result = self.validate_question(question)
            batch_results["question_results"].append(result)
            
            if result.is_valid:
                batch_results["valid_questions"] += 1
            else:
                batch_results["invalid_questions"] += 1
            
            total_score += result.score
            
            # Collect all errors, warnings, and suggestions
            all_errors.update(result.errors)
            all_warnings.update(result.warnings)
            all_suggestions.update(result.suggestions)
        
        # Calculate averages
        batch_results["average_score"] = total_score / len(questions)
        batch_results["overall_errors"] = list(all_errors)
        batch_results["overall_warnings"] = list(all_warnings)
        batch_results["overall_suggestions"] = list(all_suggestions)
        
        return batch_results
    
    def validate_latex_document(self, latex_content: str) -> ValidationResult:
        """
        Validate a complete LaTeX document.
        
        Args:
            latex_content: Complete LaTeX document
            
        Returns:
            Validation result
        """
        try:
            errors = []
            warnings = []
            suggestions = []
            score = 0.0
            
            # Check document structure
            if "\\documentclass" in latex_content:
                score += 0.2
            else:
                errors.append("Missing document class")
            
            if "\\begin{document}" in latex_content:
                score += 0.2
            else:
                errors.append("Missing document begin")
            
            if "\\end{document}" in latex_content:
                score += 0.2
            else:
                errors.append("Missing document end")
            
            # Check for required packages
            required_packages = ["amsmath", "amssymb"]
            for package in required_packages:
                if f"\\usepackage{{{package}}}" in latex_content:
                    score += 0.1
                else:
                    warnings.append(f"Missing package: {package}")
            
            # Check for mathematical content
            math_patterns = [r'\$[^$]*\$', r'\\[a-z]+', r'\\text\{[^}]*\}']
            math_content = 0
            for pattern in math_patterns:
                if re.search(pattern, latex_content):
                    math_content += 1
            
            if math_content >= 2:
                score += 0.3
            else:
                warnings.append("Limited mathematical content")
            
            # Check for common LaTeX issues
            if latex_content.count('$') % 2 != 0:
                errors.append("Unmatched dollar signs in document")
            
            if '\\text{' in latex_content and '\\text{' not in latex_content:
                warnings.append("Inconsistent text formatting")
            
            is_valid = score >= 0.7 and len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                score=score,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                metadata={
                    "document_length": len(latex_content),
                    "math_patterns_found": math_content,
                    "has_required_packages": all(f"\\usepackage{{{pkg}}}" in latex_content for pkg in required_packages)
                }
            )
            
        except Exception as e:
            logger.error(f"LaTeX document validation error: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Document validation failed: {str(e)}"],
                warnings=[],
                suggestions=[],
                metadata={}
            )
    
    def generate_validation_report(self, questions: List[Dict[str, Any]], 
                                 latex_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            questions: List of questions
            latex_content: Optional LaTeX document content
            
        Returns:
            Validation report
        """
        report = {
            "summary": {},
            "question_validation": {},
            "latex_validation": {},
            "recommendations": []
        }
        
        # Question validation
        question_validation = self.validate_question_batch(questions)
        report["question_validation"] = question_validation
        
        # Summary statistics
        report["summary"] = {
            "total_questions": question_validation["total_questions"],
            "valid_questions": question_validation["valid_questions"],
            "invalid_questions": question_validation["invalid_questions"],
            "validation_rate": question_validation["valid_questions"] / max(question_validation["total_questions"], 1),
            "average_score": question_validation["average_score"],
        }
        
        # LaTeX validation
        if latex_content:
            latex_validation = self.validate_latex_document(latex_content)
            report["latex_validation"] = {
                "is_valid": latex_validation.is_valid,
                "score": latex_validation.score,
                "errors": latex_validation.errors,
                "warnings": latex_validation.warnings,
                "suggestions": latex_validation.suggestions,
                "metadata": latex_validation.metadata
            }
        
        # Generate recommendations
        recommendations = []
        
        if question_validation["invalid_questions"] > 0:
            recommendations.append(f"Review {question_validation['invalid_questions']} invalid questions")
        
        if question_validation["average_score"] < 0.8:
            recommendations.append("Improve overall question quality")
        
        if question_validation["overall_errors"]:
            recommendations.append("Fix validation errors")
        
        if question_validation["overall_warnings"]:
            recommendations.append("Address validation warnings")
        
        if latex_content and not report["latex_validation"]["is_valid"]:
            recommendations.append("Fix LaTeX document issues")
        
        report["recommendations"] = recommendations
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], filename: str, 
                             output_dir: Optional[str] = None) -> str:
        """Save validation report to file."""
        try:
            output_path = Path(output_dir or config.output_dir) / "validation_reports" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise ValidationError(
                f"Failed to save validation report: {str(e)}",
                context={"filename": filename}
            ) 