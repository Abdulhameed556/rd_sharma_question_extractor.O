"""
LaTeX converter for RD Sharma Question Extractor.

This module provides LaTeX formatting capabilities for mathematical
expressions and question content.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError
from ..config import config

logger = get_logger(__name__)


class LaTeXConverter:
    """Converts mathematical expressions to LaTeX format."""
    
    def __init__(self):
        """Initialize LaTeX converter with mathematical symbol mappings."""
        # Mathematical symbol mappings
        self.math_symbols = {
            # Basic operations
            '+': '+',
            '-': '-',
            '×': '\\times',
            '÷': '\\div',
            '=': '=',
            '<': '<',
            '>': '>',
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '±': '\\pm',
            
            # Greek letters
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'ε': '\\epsilon',
            'θ': '\\theta',
            'λ': '\\lambda',
            'μ': '\\mu',
            'π': '\\pi',
            'σ': '\\sigma',
            'φ': '\\phi',
            'ψ': '\\psi',
            'ω': '\\omega',
            
            # Mathematical functions
            'sin': '\\sin',
            'cos': '\\cos',
            'tan': '\\tan',
            'log': '\\log',
            'ln': '\\ln',
            'exp': '\\exp',
            'sqrt': '\\sqrt',
            'abs': '|',
            
            # Set operations
            '∩': '\\cap',
            '∪': '\\cup',
            '∈': '\\in',
            '∉': '\\notin',
            '⊂': '\\subset',
            '⊃': '\\supset',
            '⊆': '\\subseteq',
            '⊇': '\\supseteq',
            
            # Other symbols
            '∞': '\\infty',
            '∑': '\\sum',
            '∫': '\\int',
            '∂': '\\partial',
            '∆': '\\Delta',
            '∇': '\\nabla',
        }
        
        # Probability patterns
        self.probability_patterns = [
            (r'\bP\(([^)]+)\)', r'P(\\text{\1})'),  # P(event) -> P(\text{event})
            (r'\bP\(([^)]+)\|([^)]+)\)', r'P(\\text{\1} | \\text{\2})'),  # P(A|B) -> P(\text{A} | \text{B})
        ]
        
        # Number patterns
        self.number_patterns = [
            (r'\b(\d+)\b', r'$\1$'),  # Standalone numbers
            (r'\b(\d+)\s*([+\-×÷=<>≤≥≠±])\s*(\d+)\b', r'$\1 \2 \3$'),  # Arithmetic expressions
        ]
        
        logger.info("LaTeX converter initialized")
    
    def convert_question_to_latex(self, question_text: str) -> str:
        """
        Convert question text to LaTeX format.
        
        Args:
            question_text: Raw question text
            
        Returns:
            LaTeX formatted question text
        """
        try:
            # Step 1: Convert mathematical expressions
            latex_text = self._convert_math_expressions(question_text)
            
            # Step 2: Convert probability expressions
            latex_text = self._convert_probability_expressions(latex_text)
            
            # Step 3: Convert numbers
            latex_text = self._convert_numbers(latex_text)
            
            # Step 4: Clean up formatting
            latex_text = self._cleanup_latex(latex_text)
            
            return latex_text
            
        except Exception as e:
            raise ValidationError(
                f"Failed to convert question to LaTeX: {str(e)}",
                context={"question_text": question_text[:100]}
            )
    
    def _convert_math_expressions(self, text: str) -> str:
        """Convert mathematical expressions to LaTeX."""
        # Replace mathematical symbols
        for symbol, latex_symbol in self.math_symbols.items():
            text = text.replace(symbol, latex_symbol)
        
        # Convert function calls
        function_patterns = [
            (r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(([^)]+)\)', r'\\\1(\2)'),
        ]
        
        for pattern, replacement in function_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _convert_probability_expressions(self, text: str) -> str:
        """Convert probability expressions to LaTeX."""
        for pattern, replacement in self.probability_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Handle complex probability expressions
        text = re.sub(r'\bP\(([^)]+)\)', r'$P(\\text{\1})$', text)
        
        return text
    
    def _convert_numbers(self, text: str) -> str:
        """Convert numbers to LaTeX format."""
        # Convert standalone numbers (but not in existing LaTeX)
        def replace_number(match):
            number = match.group(1)
            # Check if already in LaTeX
            start_pos = match.start()
            if start_pos > 0 and text[start_pos - 1] == '$':
                return match.group(0)  # Already in LaTeX
            return f'${number}$'
        
        text = re.sub(r'\b(\d+)\b', replace_number, text)
        
        return text
    
    def _cleanup_latex(self, text: str) -> str:
        """Clean up LaTeX formatting."""
        # Remove double dollar signs
        text = re.sub(r'\$\$+', '$', text)
        
        # Fix spacing around LaTeX delimiters
        text = re.sub(r'\s+\$', '$', text)
        text = re.sub(r'\$\s+', '$', text)
        
        # Fix common LaTeX issues
        text = text.replace('\\text{', '\\text{')
        text = text.replace('\\text{', '\\text{')
        
        return text.strip()
    
    def generate_latex_document(self, questions: List[Dict[str, Any]], 
                              chapter: int, topic: str, topic_name: str) -> str:
        """
        Generate a complete LaTeX document with questions.
        
        Args:
            questions: List of question dictionaries
            chapter: Chapter number
            topic: Topic identifier
            topic_name: Topic name
            
        Returns:
            Complete LaTeX document
        """
        try:
            # Document preamble
            latex_doc = self._generate_preamble(chapter, topic_name)
            
            # Document body
            latex_doc += self._generate_body(questions, chapter, topic, topic_name)
            
            # Document end
            latex_doc += "\\end{document}\n"
            
            return latex_doc
            
        except Exception as e:
            raise ValidationError(
                f"Failed to generate LaTeX document: {str(e)}",
                context={"question_count": len(questions)}
            )
    
    def _generate_preamble(self, chapter: int, topic_name: str) -> str:
        """Generate LaTeX document preamble."""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\usepackage{{enumitem}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}

% Page setup
\\geometry{{margin=1in}}

% Title
\\title{{Chapter {chapter}: {topic_name} \\\\ Questions}}
\\author{{RD Sharma Question Extractor}}
\\date{{\\today}}

% Custom commands
\\newcommand{{\\prob}}[1]{{$P(\\text{{#1}})$}}
\\newcommand{{\\condprob}}[2]{{$P(\\text{{#1}} | \\text{{#2}})$}}

\\begin{{document}}
\\maketitle

"""
    
    def _generate_body(self, questions: List[Dict[str, Any]], 
                      chapter: int, topic: str, topic_name: str) -> str:
        """Generate LaTeX document body."""
        body = f"\\section{{Chapter {chapter}: {topic_name}}}\n\n"
        
        if not questions:
            body += "\\textit{No questions found for this topic.}\n\n"
            return body
        
        # Group questions by source
        questions_by_source = {}
        for question in questions:
            source = question.get("source", "Unknown")
            if source not in questions_by_source:
                questions_by_source[source] = []
            questions_by_source[source].append(question)
        
        # Generate sections for each source
        for source, source_questions in questions_by_source.items():
            body += f"\\subsection{{{source}}}\n\n"
            body += "\\begin{enumerate}[label=\\arabic*.]\n"
            
            for question in source_questions:
                question_text = question.get("question_text", "")
                question_number = question.get("question_number", "")
                
                # Convert to LaTeX
                latex_question = self.convert_question_to_latex(question_text)
                
                # Add question
                if question_number and question_number != "1":
                    body += f"\\item[{question_number}] {latex_question}\n"
                else:
                    body += f"\\item {latex_question}\n"
            
            body += "\\end{enumerate}\n\n"
        
        return body
    
    def validate_latex(self, latex_text: str) -> Dict[str, Any]:
        """
        Validate LaTeX syntax and formatting.
        
        Args:
            latex_text: LaTeX text to validate
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Check for unmatched delimiters
            dollar_count = latex_text.count('$')
            if dollar_count % 2 != 0:
                validation["errors"].append("Unmatched dollar signs in LaTeX")
                validation["is_valid"] = False
            
            # Check for common LaTeX errors
            if '\\text{' in latex_text and '\\text{' not in latex_text:
                validation["warnings"].append("Inconsistent text formatting")
            
            # Check for mathematical symbols
            math_symbols = ['\\times', '\\div', '\\leq', '\\geq', '\\neq', '\\pm']
            for symbol in math_symbols:
                if symbol in latex_text:
                    validation["suggestions"].append(f"Mathematical symbol used: {symbol}")
            
            # Check for probability expressions
            if 'P(' in latex_text:
                validation["suggestions"].append("Probability expressions detected")
            
            # Check for numbers
            number_pattern = r'\$\d+\$'
            numbers = re.findall(number_pattern, latex_text)
            if numbers:
                validation["suggestions"].append(f"Numbers formatted: {len(numbers)} found")
            
        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["is_valid"] = False
        
        return validation
    
    def convert_batch_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert a batch of questions to LaTeX format.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of questions with LaTeX formatting
        """
        converted_questions = []
        
        for question in questions:
            try:
                # Convert question text
                original_text = question.get("question_text", "")
                latex_text = self.convert_question_to_latex(original_text)
                
                # Create converted question
                converted_question = question.copy()
                converted_question["question_text"] = latex_text
                converted_question["original_text"] = original_text
                converted_question["latex_converted"] = True
                
                # Validate LaTeX
                validation = self.validate_latex(latex_text)
                converted_question["latex_validation"] = validation
                
                converted_questions.append(converted_question)
                
            except Exception as e:
                logger.warning(f"Failed to convert question: {e}")
                # Keep original question with error flag
                question["latex_converted"] = False
                question["latex_error"] = str(e)
                converted_questions.append(question)
        
        return converted_questions
    
    def generate_markdown_with_latex(self, questions: List[Dict[str, Any]], 
                                   chapter: int, topic: str, topic_name: str) -> str:
        """
        Generate Markdown with LaTeX for rendering.
        
        Args:
            questions: List of question dictionaries
            chapter: Chapter number
            topic: Topic identifier
            topic_name: Topic name
            
        Returns:
            Markdown content with LaTeX
        """
        markdown = f"# Chapter {chapter}: {topic_name}\n\n"
        markdown += f"**Topic:** {topic}\n\n"
        markdown += f"**Total Questions:** {len(questions)}\n\n"
        
        if not questions:
            markdown += "*No questions found for this topic.*\n\n"
            return markdown
        
        # Group by source
        questions_by_source = {}
        for question in questions:
            source = question.get("source", "Unknown")
            if source not in questions_by_source:
                questions_by_source[source] = []
            questions_by_source[source].append(question)
        
        # Generate sections
        for source, source_questions in questions_by_source.items():
            markdown += f"## {source}\n\n"
            
            for i, question in enumerate(source_questions, 1):
                question_text = question.get("question_text", "")
                question_number = question.get("question_number", str(i))
                
                markdown += f"**{question_number}.** {question_text}\n\n"
        
        return markdown
    
    def save_latex_file(self, latex_content: str, filename: str, output_dir: Optional[str] = None):
        """Save LaTeX content to file."""
        try:
            output_path = Path(output_dir or config.latex_output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            logger.info(f"LaTeX file saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise ValidationError(
                f"Failed to save LaTeX file: {str(e)}",
                context={"filename": filename}
            )
    
    def get_conversion_statistics(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about LaTeX conversion."""
        stats = {
            "total_questions": len(questions),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "validation_errors": 0,
            "validation_warnings": 0,
        }
        
        for question in questions:
            if question.get("latex_converted", False):
                stats["successful_conversions"] += 1
                
                validation = question.get("latex_validation", {})
                stats["validation_errors"] += len(validation.get("errors", []))
                stats["validation_warnings"] += len(validation.get("warnings", []))
            else:
                stats["failed_conversions"] += 1
        
        return stats 