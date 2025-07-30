"""
LaTeX renderer utilities for RD Sharma Question Extractor.

This module provides LaTeX rendering, validation, and mathematical expression
checking capabilities for the question extraction system.
"""

import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np

from .logger import get_logger
from .exceptions import ValidationError
from ..config import config

logger = get_logger(__name__)

# Configure matplotlib for LaTeX rendering
rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    'font.family': 'serif',
    'font.size': 12
})


class LaTeXRenderer:
    """Handles LaTeX rendering and validation for mathematical expressions."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize LaTeX renderer.

        Args:
            output_dir: Directory for rendered output files
        """
        self.output_dir = Path(output_dir or config.latex_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mathematical expression patterns
        self.math_patterns = {
            'numbers': r'\$(\d+(?:\.\d+)?)\$',
            'fractions': r'\\frac\{[^}]+\}\{[^}]+\}',
            'probability': r'P\([^)]+\)',
            'conditional_prob': r'P\([^|]+\|[^)]+\)',
            'sets': r'[A-Z]\s*[∩∪⊂⊃∈∉]',
            'integrals': r'\\int[^}]*',
            'summations': r'\\sum[^}]*',
            'roots': r'\\sqrt[^}]*',
            'subscripts': r'[a-zA-Z]_\{[^}]+\}',
            'superscripts': r'[a-zA-Z]\^\{[^}]+\}'
        }
        
        # Common LaTeX syntax errors
        self.syntax_errors = {
            'unmatched_braces': r'\{[^{}]*$|^[^{}]*\}',
            'unmatched_dollars': r'\$[^$]*$|^[^$]*\$',
            'invalid_commands': r'\\[a-zA-Z]+\{[^}]*$',
            'missing_backslash': r'(?<!\\)[a-zA-Z]+\{[^}]*\}',
            'invalid_math_mode': r'\$[^$]*[a-zA-Z]+\{[^}]*\}[^$]*\$'
        }
        
        logger.info(f"LaTeX renderer initialized with output directory: {self.output_dir}")

    def validate_latex_syntax(self, latex_content: str) -> Dict[str, any]:
        """
        Validate LaTeX syntax and mathematical expressions.

        Args:
            latex_content: LaTeX content to validate

        Returns:
            Validation result dictionary
        """
        try:
            errors = []
            warnings = []
            suggestions = []
            
            # Check for basic syntax errors
            for error_type, pattern in self.syntax_errors.items():
                matches = re.findall(pattern, latex_content, re.MULTILINE)
                if matches:
                    errors.append({
                        'type': error_type,
                        'message': f'Found {len(matches)} {error_type}',
                        'matches': matches[:5]  # Limit to first 5 matches
                    })
            
            # Check for mathematical expression patterns
            math_score = 0
            total_patterns = len(self.math_patterns)
            
            for pattern_name, pattern in self.math_patterns.items():
                matches = re.findall(pattern, latex_content)
                if matches:
                    math_score += 1
                    logger.debug(f"Found {len(matches)} {pattern_name} expressions")
            
            # Check for common LaTeX issues
            if '\\text{' in latex_content and '\\text{' not in latex_content:
                warnings.append({
                    'type': 'text_command',
                    'message': 'Inconsistent use of \\text{} command'
                })
            
            if latex_content.count('$') % 2 != 0:
                errors.append({
                    'type': 'unmatched_dollars',
                    'message': 'Unmatched dollar signs in LaTeX content'
                })
            
            if latex_content.count('{') != latex_content.count('}'):
                errors.append({
                    'type': 'unmatched_braces',
                    'message': 'Unmatched braces in LaTeX content'
                })
            
            # Calculate overall score
            syntax_score = max(0, 100 - len(errors) * 20 - len(warnings) * 10)
            math_score = (math_score / total_patterns) * 100
            
            # Generate suggestions
            if math_score < 50:
                suggestions.append("Consider adding more mathematical expressions")
            
            if '\\frac' not in latex_content and '/' in latex_content:
                suggestions.append("Consider converting fractions to LaTeX format")
            
            if 'P(' not in latex_content and 'probability' in latex_content.lower():
                suggestions.append("Consider using LaTeX probability notation P()")
            
            return {
                'is_valid': len(errors) == 0,
                'syntax_score': syntax_score,
                'math_score': math_score,
                'errors': errors,
                'warnings': warnings,
                'suggestions': suggestions,
                'total_expressions': sum(len(re.findall(pattern, latex_content)) 
                                       for pattern in self.math_patterns.values())
            }
            
        except Exception as e:
            logger.error(f"Error validating LaTeX syntax: {e}")
            return {
                'is_valid': False,
                'syntax_score': 0,
                'math_score': 0,
                'errors': [{'type': 'validation_error', 'message': str(e)}],
                'warnings': [],
                'suggestions': []
            }

    def render_latex_expression(self, latex_expr: str, filename: Optional[str] = None) -> str:
        """
        Render a LaTeX expression to an image.

        Args:
            latex_expr: LaTeX expression to render
            filename: Optional output filename

        Returns:
            Path to rendered image file
        """
        try:
            # Clean up LaTeX expression
            latex_expr = self._clean_latex_expression(latex_expr)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, f'${latex_expr}$', 
                   fontsize=16, ha='center', va='center',
                   transform=ax.transAxes)
            
            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Generate filename if not provided
            if not filename:
                safe_expr = re.sub(r'[^a-zA-Z0-9]', '_', latex_expr)[:30]
                filename = f"latex_{safe_expr}.png"
            
            output_path = self.output_dir / filename
            
            # Save image
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"LaTeX expression rendered: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error rendering LaTeX expression: {e}")
            raise ValidationError(f"Failed to render LaTeX expression: {str(e)}")

    def render_question_batch(self, questions: List[Dict[str, str]], 
                            output_filename: str = "questions_rendered.png") -> str:
        """
        Render a batch of questions as a single image.

        Args:
            questions: List of question dictionaries
            output_filename: Output filename

        Returns:
            Path to rendered image file
        """
        try:
            n_questions = len(questions)
            fig, axes = plt.subplots(n_questions, 1, figsize=(10, 2 * n_questions))
            
            if n_questions == 1:
                axes = [axes]
            
            for i, question in enumerate(questions):
                question_text = question.get('question_text', '')
                question_number = question.get('question_number', f'{i+1}')
                
                # Clean and format question text
                formatted_text = self._format_question_for_display(question_text)
                
                axes[i].text(0.05, 0.5, f"{question_number}. {formatted_text}", 
                           fontsize=12, ha='left', va='center',
                           transform=axes[i].transAxes, wrap=True)
                axes[i].set_xlim(0, 1)
                axes[i].set_ylim(0, 1)
                axes[i].axis('off')
            
            output_path = self.output_dir / output_filename
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Question batch rendered: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error rendering question batch: {e}")
            raise ValidationError(f"Failed to render question batch: {str(e)}")

    def create_latex_document(self, questions: List[Dict[str, str]], 
                            title: str = "Extracted Questions",
                            author: str = "RD Sharma Question Extractor") -> str:
        """
        Create a complete LaTeX document with questions.

        Args:
            questions: List of question dictionaries
            title: Document title
            author: Document author

        Returns:
            Path to generated LaTeX file
        """
        try:
            latex_content = self._generate_latex_preamble(title, author)
            latex_content += self._generate_latex_body(questions)
            latex_content += r'\end{document}'
            
            # Validate the complete document
            validation_result = self.validate_latex_syntax(latex_content)
            
            if not validation_result['is_valid']:
                logger.warning(f"LaTeX document has validation issues: {validation_result['errors']}")
            
            # Save LaTeX file
            output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.tex"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            logger.info(f"LaTeX document created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating LaTeX document: {e}")
            raise ValidationError(f"Failed to create LaTeX document: {str(e)}")

    def compile_latex_document(self, tex_file_path: str) -> str:
        """
        Compile a LaTeX document to PDF.

        Args:
            tex_file_path: Path to LaTeX file

        Returns:
            Path to compiled PDF file
        """
        try:
            tex_path = Path(tex_file_path)
            if not tex_path.exists():
                raise ValidationError(f"LaTeX file not found: {tex_file_path}")
            
            # Change to LaTeX file directory
            original_dir = Path.cwd()
            subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path.name], 
                           capture_output=True, text=True, timeout=30)
            
            # Return to original directory
            subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path.name], 
                           capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"LaTeX compilation failed: {result.stderr}")
                raise ValidationError(f"LaTeX compilation failed: {result.stderr}")
            
            # Get PDF path
            pdf_path = tex_path.with_suffix('.pdf')
            
            if not pdf_path.exists():
                raise ValidationError("PDF file was not generated")
            
            logger.info(f"LaTeX document compiled: {pdf_path}")
            return str(pdf_path)
            
        except subprocess.TimeoutExpired:
            raise ValidationError("LaTeX compilation timed out")
        except Exception as e:
            logger.error(f"Error compiling LaTeX document: {e}")
            raise ValidationError(f"Failed to compile LaTeX document: {str(e)}")

    def generate_preview_report(self, questions: List[Dict[str, str]], 
                              output_filename: str = "preview_report.html") -> str:
        """
        Generate an HTML preview report of extracted questions.

        Args:
            questions: List of question dictionaries
            output_filename: Output filename

        Returns:
            Path to generated HTML file
        """
        try:
            html_content = self._generate_html_header()
            
            for i, question in enumerate(questions):
                question_text = question.get('question_text', '')
                question_number = question.get('question_number', f'{i+1}')
                source = question.get('source', 'Unknown')
                
                # Validate LaTeX in question
                validation = self.validate_latex_syntax(question_text)
                
                html_content += f"""
                <div class="question">
                    <h3>Question {question_number}</h3>
                    <p class="source">Source: {source}</p>
                    <div class="latex-content">
                        {question_text}
                    </div>
                    <div class="validation">
                        <span class="score">LaTeX Score: {validation['syntax_score']:.1f}%</span>
                        <span class="math-score">Math Score: {validation['math_score']:.1f}%</span>
                        {f'<span class="error">Errors: {len(validation["errors"])}</span>' if validation["errors"] else ''}
                    </div>
                </div>
                """
            
            html_content += self._generate_html_footer()
            
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Preview report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating preview report: {e}")
            raise ValidationError(f"Failed to generate preview report: {str(e)}")

    def _clean_latex_expression(self, latex_expr: str) -> str:
        """Clean and normalize LaTeX expression."""
        # Remove extra whitespace
        latex_expr = re.sub(r'\s+', ' ', latex_expr.strip())
        
        # Fix common LaTeX issues
        latex_expr = re.sub(r'\\text\{([^}]+)\}', r'\1', latex_expr)  # Simplify text commands
        latex_expr = re.sub(r'\$\s*([^$]+)\s*\$', r'$\1$', latex_expr)  # Clean dollar signs
        
        return latex_expr

    def _format_question_for_display(self, question_text: str) -> str:
        """Format question text for display rendering."""
        # Convert LaTeX to display format
        formatted = question_text.replace('$', '')  # Remove dollar signs for display
        formatted = re.sub(r'\\text\{([^}]+)\}', r'\1', formatted)  # Simplify text commands
        
        return formatted

    def _generate_latex_preamble(self, title: str, author: str) -> str:
        """Generate LaTeX document preamble."""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\usepackage{{enumitem}}

\\geometry{{margin=1in}}

\\title{{{title}}}
\\author{{{author}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Extracted Questions}}

\\begin{{enumerate}}
"""

    def _generate_latex_body(self, questions: List[Dict[str, str]]) -> str:
        """Generate LaTeX document body with questions."""
        body = ""
        
        for question in questions:
            question_text = question.get('question_text', '')
            # Clean up question text for LaTeX
            cleaned_text = question_text.replace('\\', '\\\\')
            body += f"\\item {cleaned_text}\n\n"
        
        body += "\\end{enumerate}\n"
        return body

    def _generate_html_header(self) -> str:
        """Generate HTML header for preview report."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>RD Sharma Question Extractor - Preview Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .question { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .source { color: #666; font-style: italic; }
        .latex-content { font-family: 'Courier New', monospace; background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .validation { margin-top: 10px; }
        .score { color: green; margin-right: 15px; }
        .math-score { color: blue; margin-right: 15px; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>RD Sharma Question Extractor - Preview Report</h1>
"""

    def _generate_html_footer(self) -> str:
        """Generate HTML footer for preview report."""
        return """
</body>
</html>
"""

    def get_rendering_statistics(self) -> Dict[str, any]:
        """Get statistics about LaTeX rendering operations."""
        try:
            rendered_files = list(self.output_dir.glob("*.png"))
            tex_files = list(self.output_dir.glob("*.tex"))
            pdf_files = list(self.output_dir.glob("*.pdf"))
            
            return {
                'total_rendered_images': len(rendered_files),
                'total_tex_files': len(tex_files),
                'total_pdf_files': len(pdf_files),
                'output_directory': str(self.output_dir),
                'last_operation': 'LaTeX rendering statistics'
            }
            
        except Exception as e:
            logger.error(f"Error getting rendering statistics: {e}")
            return {'error': str(e)} 