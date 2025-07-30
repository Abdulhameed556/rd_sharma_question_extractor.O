"""
Prompt templates for RD Sharma Question Extractor.

This module provides optimized prompt templates for the Groq Llama-4-Maverick-17B model
for question extraction and LaTeX formatting.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from config import config


@dataclass
class PromptTemplate:
    """Prompt template with metadata."""
    name: str
    template: str
    description: str
    version: str
    parameters: List[str]
    expected_output: str


class PromptTemplates:
    """Manages prompt templates for different LLM tasks."""

    def __init__(self, config):
        """Initialize prompt templates."""
        self.config = config
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            "question_extraction": PromptTemplate(
                name="question_extraction",
                template=self._get_question_extraction_prompt(),
                description="Extract mathematical questions from textbook content",
                version="2.0",
                parameters=["content", "chapter", "topic"],
                expected_output="JSON array of questions with LaTeX formatting"
            ),
            "latex_formatting": PromptTemplate(
                name="latex_formatting",
                template=self._get_latex_formatting_prompt(),
                description="Convert raw question text to LaTeX format",
                version="1.5",
                parameters=["question_text"],
                expected_output="LaTeX formatted question text"
            ),
            "content_validation": PromptTemplate(
                name="content_validation",
                template=self._get_content_validation_prompt(),
                description="Validate extracted content quality",
                version="1.0",
                parameters=["content", "content_type"],
                expected_output="JSON validation result"
            ),
            "mathematical_correction": PromptTemplate(
                name="mathematical_correction",
                template=self._get_mathematical_correction_prompt(),
                description="Correct mathematical expressions and notation",
                version="1.2",
                parameters=["expression"],
                expected_output="Corrected mathematical expression"
            ),
            "question_classification": PromptTemplate(
                name="question_classification",
                template=self._get_question_classification_prompt(),
                description="Classify question types and difficulty",
                version="1.0",
                parameters=["question_text"],
                expected_output="JSON classification result"
            )
        }

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with given parameters."""
        template = self.get_template(template_name)
        
        try:
            formatted_prompt = template.template.format(**kwargs)
            return formatted_prompt
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

    def _get_question_extraction_prompt(self) -> str:
        """Get the main question extraction prompt."""
        return """You are an expert mathematical content extractor specializing in LaTeX formatting for academic publications.

CRITICAL MISSION: Extract ONLY questions from textbook content and convert ALL numerical and mathematical content to professional LaTeX format.

MANDATORY LATEX RULES - FOLLOW EXACTLY:
- ALL numbers must be in LaTeX: "4 red balls" → "$4$ red balls"
- ALL mathematical expressions: "P(A|B)" → "$P(A|B)$"
- ALL fractions: "1/2" → "$\\frac{{1}}{{2}}$"
- ALL probability expressions: "P(sum = 8)" → "$P(\\text{{sum}} = 8)$"
- ALL sets/intersections: "P(A∩B)" → "$P(A \\cap B)$"
- ALL conditional notation: "given that" → "given that" (keep text, but math in LaTeX)

ADVANCED PROBABILITY NOTATION:
- Events: "both balls are red" → "$P(\\text{{both balls are red}})$"
- Conditions: "given that first throw shows even" → "given that $P(\\text{{first throw is even}})$"
- Complex events: "both cards are aces" → "$P(\\text{{both aces}})$"
- Set descriptions: "at least one ace" → "$P(\\text{{at least one ace}})$"
- Game combinations: "plays exactly one game" → "$P(\\text{{exactly one game}})$"

TASK: Extract ONLY questions (ignore theory, explanations, solutions) and apply AGGRESSIVE LaTeX formatting.

CONTEXT: Chapter {chapter}, Topic {topic}

SAMPLE TEXT:
{content}

REQUIRED OUTPUT FORMAT: json
[
  {{
    "question_number": "Illustration 1",
    "question_text": "A bag contains $4$ red balls and $6$ black balls. Two balls are drawn at random without replacement. Find $P(\\text{{both balls are red}})$.",
    "source": "Illustration"
  }},
  {{
    "question_number": "1",
    "question_text": "A die is thrown twice. Find $P(\\text{{sum}} = 8 | \\text{{first throw is even}})$.",
    "source": "Exercise {topic}"
  }}
]

LATEX FORMATTING EXAMPLES - FOLLOW THESE PATTERNS:
- Numbers: $4$, $6$, $8$, $30$, $52$, $60$
- Basic Probability: $P(\\text{{sum}} = 8)$, $P(\\text{{both aces}})$
- Conditional Probability: $P(A|B)$, $P(\\text{{cricket}}|\\text{{football}})$, $P(\\text{{sum}} = 8 | \\text{{first throw is even}})$
- Complex Events: $P(\\text{{both balls are red}})$, $P(\\text{{exactly one game}})$, $P(\\text{{at least one ace}})$
- Sets: $P(A \\cap B)$, $P(A \\cup B)$
- Text in math: Always use $P(\\text{{descriptive event}})$ format

VERIFICATION CHECKLIST - Your output MUST have:
✓ ALL numbers wrapped in $ $ delimiters
✓ ALL probability expressions in proper LaTeX format: $P(\\text{{event}})$
✓ ALL conditional probabilities: $P(\\text{{A}} | \\text{{B}})$
✓ ALL complex probability statements converted to mathematical notation
✓ NO plain text numbers (4 → $4$)
✓ NO plain text probability statements ("find the probability that..." → "find $P(\\text{{...}})$")
✓ Valid JSON structure
✓ Only questions extracted (no theory/solutions)

CRITICAL: Every probability statement must be in formal mathematical notation. Convert all "find the probability that X" to "find $P(\\text{{X}})$" format.

CRITICAL: If you see ANY plain text numbers in your output, you have FAILED the task. Every single number must be in LaTeX format.

Generate the JSON now with PERFECT LaTeX formatting:"""

    def _get_latex_formatting_prompt(self) -> str:
        """Get LaTeX formatting prompt."""
        return """You are a LaTeX formatting expert specializing in mathematical expressions.

TASK: Convert the following question text to proper LaTeX format, ensuring all mathematical expressions, numbers, and symbols are correctly formatted.

QUESTION TEXT:
{question_text}

REQUIREMENTS:
1. All numbers must be in LaTeX math mode: 4 → $4$
2. All mathematical expressions must be properly formatted
3. All probability expressions must use $P(\\text{{event}})$ format
4. All fractions must use $\\frac{{numerator}}{{denominator}}$ format
5. All conditional probabilities must use $P(A|B)$ format
6. All set operations must use proper LaTeX symbols ($\\cap$, $\\cup$, $\\in$, etc.)

OUTPUT: Return only the LaTeX formatted question text.

EXAMPLE:
Input: "Find the probability that both balls are red given that the first ball is red, where there are 4 red balls and 6 black balls."
Output: "Find $P(\\text{{both balls are red}} | \\text{{first ball is red}})$, where there are $4$ red balls and $6$ black balls."

Format the question now:"""

    def _get_content_validation_prompt(self) -> str:
        """Get content validation prompt."""
        return """You are a quality assurance expert for mathematical content extraction.

TASK: Validate the following extracted content for quality, completeness, and correctness.

CONTENT TYPE: {content_type}

CONTENT:
{content}

VALIDATION CRITERIA:
1. Completeness: Does the content contain all necessary information?
2. Accuracy: Are mathematical expressions correct?
3. LaTeX Formatting: Are all numbers and expressions properly formatted?
4. Clarity: Is the content clear and understandable?
5. Relevance: Is the content relevant to the requested topic?

OUTPUT FORMAT: JSON
{{
    "is_valid": true/false,
    "score": 0-100,
    "completeness": 0-100,
    "accuracy": 0-100,
    "latex_quality": 0-100,
    "clarity": 0-100,
    "relevance": 0-100,
    "errors": ["list of specific errors"],
    "warnings": ["list of warnings"],
    "suggestions": ["list of improvement suggestions"]
}}

Provide validation result:"""

    def _get_mathematical_correction_prompt(self) -> str:
        """Get mathematical correction prompt."""
        return """You are a mathematical notation expert.

TASK: Correct the following mathematical expression to use proper mathematical notation and LaTeX formatting.

EXPRESSION:
{expression}

CORRECTION GUIDELINES:
1. Fix common OCR errors (I → |, 1 → l, 0 → O, etc.)
2. Ensure proper mathematical notation
3. Convert to LaTeX format where appropriate
4. Fix spacing and formatting issues
5. Ensure consistency in notation

OUTPUT: Return only the corrected expression.

EXAMPLE:
Input: "P(AIB)" (OCR error)
Output: "P(A|B)"

Input: "4 red bal1s" (OCR error)
Output: "4 red balls"

Correct the expression:"""

    def _get_question_classification_prompt(self) -> str:
        """Get question classification prompt."""
        return """You are an expert in mathematical question classification.

TASK: Classify the following mathematical question by type, difficulty, and topic.

QUESTION:
{question_text}

CLASSIFICATION CRITERIA:
1. Question Type: illustration, exercise, practice, theory, application
2. Difficulty Level: easy, medium, hard, expert
3. Mathematical Topic: probability, algebra, calculus, geometry, etc.
4. Problem Type: calculation, proof, application, analysis
5. Required Skills: basic arithmetic, probability concepts, conditional probability, etc.

OUTPUT FORMAT: JSON
{{
    "question_type": "illustration|exercise|practice|theory|application",
    "difficulty": "easy|medium|hard|expert",
    "topic": "specific mathematical topic",
    "problem_type": "calculation|proof|application|analysis",
    "required_skills": ["list of required mathematical skills"],
    "estimated_solving_time": "time in minutes",
    "confidence": 0-100
}}

Classify the question:"""

    def get_prompt_metadata(self, template_name: str) -> Dict:
        """Get metadata for a prompt template."""
        template = self.get_template(template_name)
        return {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "parameters": template.parameters,
            "expected_output": template.expected_output
        }

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def validate_parameters(self, template_name: str, **kwargs) -> bool:
        """Validate that all required parameters are provided."""
        template = self.get_template(template_name)
        provided_params = set(kwargs.keys())
        required_params = set(template.parameters)
        
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        return True

    def get_prompt_statistics(self) -> Dict:
        """Get statistics about prompt templates."""
        return {
            "total_templates": len(self.templates),
            "template_names": list(self.templates.keys()),
            "latest_version": max(template.version for template in self.templates.values()),
            "parameter_count": sum(len(template.parameters) for template in self.templates.values())
        } 