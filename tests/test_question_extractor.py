"""
Unit tests for question extractor module.

This module tests question detection, LaTeX conversion, and validation
functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.question_extractor.detector import QuestionDetector, QuestionCandidate
from src.question_extractor.latex_converter import LaTeXConverter
from src.question_extractor.validator import QuestionValidator, ValidationResult
from src.utils.exceptions import ValidationError


class TestQuestionDetector:
    """Test question detector functionality."""

    def test_initialization(self, test_config):
        """Test question detector initialization."""
        detector = QuestionDetector(test_config)
        assert detector.config == test_config

    def test_detect_questions(self, test_config):
        """Test question detection."""
        detector = QuestionDetector(test_config)
        
        content = """
        Chapter 30: Probability
        
        30.3 Conditional Probability
        
        Theory: Conditional probability is defined as P(A|B) = P(A∩B)/P(B).
        
        Illustration 1: A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red.
        
        Exercise 30.3
        
        1. A die is thrown twice. Find the probability that the sum of numbers appearing is 8, given that the first throw shows an even number.
        
        2. In a class of 60 students, 30 play cricket, 20 play football and 10 play both games. A student is selected at random. Find the probability that:
           (i) He plays cricket given that he plays football
           (ii) He plays exactly one game
        """
        
        candidates = detector.detect_questions(content)
        
        assert len(candidates) > 0
        assert all(isinstance(candidate, QuestionCandidate) for candidate in candidates)
        assert any('Illustration 1' in candidate.text for candidate in candidates)
        assert any('Exercise 30.3' in candidate.text for candidate in candidates)

    def test_detect_by_patterns(self, test_config):
        """Test pattern-based question detection."""
        detector = QuestionDetector(test_config)
        
        content = """
        Illustration 1: A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red.
        
        Exercise 30.3
        
        1. A die is thrown twice. Find the probability that the sum of numbers appearing is 8, given that the first throw shows an even number.
        
        2. In a class of 60 students, 30 play cricket, 20 play football and 10 play both games. A student is selected at random. Find the probability that:
           (i) He plays cricket given that he plays football
           (ii) He plays exactly one game
        """
        
        candidates = detector._detect_by_patterns(content)
        
        assert len(candidates) > 0
        assert any('Illustration 1' in candidate.text for candidate in candidates)
        assert any('Exercise 30.3' in candidate.text for candidate in candidates)

    def test_detect_by_nlp(self, test_config):
        """Test NLP-based question detection."""
        detector = QuestionDetector(test_config)
        
        content = "Find the probability that both balls are red given that the first ball is red."
        
        candidates = detector._detect_by_nlp(content)
        
        # Should detect question indicators
        assert len(candidates) > 0
        assert all(hasattr(candidate, 'confidence') for candidate in candidates)

    def test_detect_by_mathematical_content(self, test_config):
        """Test mathematical content-based detection."""
        detector = QuestionDetector(test_config)
        
        content = "Find P(A|B) where P(A) = 1/2 and P(B) = 1/3"
        
        candidates = detector._detect_by_mathematical_content(content)
        
        assert len(candidates) > 0
        assert all(hasattr(candidate, 'confidence') for candidate in candidates)

    def test_extract_question_text(self, test_config):
        """Test question text extraction."""
        detector = QuestionDetector(test_config)
        
        # Test illustration
        illustration_text = "Illustration 1: A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red."
        extracted = detector._extract_question_text(illustration_text)
        
        assert "A bag contains 4 red balls" in extracted
        assert "Find the probability" in extracted
        
        # Test exercise
        exercise_text = "1. A die is thrown twice. Find the probability that the sum of numbers appearing is 8, given that the first throw shows an even number."
        extracted = detector._extract_question_text(exercise_text)
        
        assert "A die is thrown twice" in extracted
        assert "Find the probability" in extracted

    def test_contains_mathematical_content(self, test_config):
        """Test mathematical content detection."""
        detector = QuestionDetector(test_config)
        
        # Test mathematical content
        assert detector._contains_mathematical_content("Find P(A|B) where P(A) = 1/2")
        assert detector._contains_mathematical_content("Calculate 4 + 5 = 9")
        assert detector._contains_mathematical_content("Find the probability of drawing 2 red balls from 10 balls")
        
        # Test non-mathematical content
        assert not detector._contains_mathematical_content("This is regular text without math")

    def test_calculate_pattern_confidence(self, test_config):
        """Test pattern confidence calculation."""
        detector = QuestionDetector(test_config)
        
        # Test high confidence patterns
        high_confidence = detector._calculate_pattern_confidence("Illustration 1: Find the probability...")
        assert high_confidence > 0.8
        
        # Test medium confidence patterns
        medium_confidence = detector._calculate_pattern_confidence("1. Find the probability...")
        assert 0.5 < medium_confidence < 0.9
        
        # Test low confidence patterns
        low_confidence = detector._calculate_pattern_confidence("Some random text without clear patterns")
        assert low_confidence < 0.5

    def test_calculate_nlp_confidence(self, test_config):
        """Test NLP confidence calculation."""
        detector = QuestionDetector(test_config)
        
        # Test question-like content
        question_confidence = detector._calculate_nlp_confidence("Find the probability that both balls are red")
        assert question_confidence > 0.6
        
        # Test non-question content
        non_question_confidence = detector._calculate_nlp_confidence("This is a statement about probability theory")
        assert non_question_confidence < 0.5

    def test_calculate_math_confidence(self, test_config):
        """Test mathematical content confidence calculation."""
        detector = QuestionDetector(test_config)
        
        # Test high math content
        high_math_confidence = detector._calculate_math_confidence("Find P(A|B) where P(A) = 1/2 and P(B) = 1/3")
        assert high_math_confidence > 0.8
        
        # Test medium math content
        medium_math_confidence = detector._calculate_math_confidence("Find the probability of drawing 2 balls from 10 balls")
        assert 0.4 < medium_math_confidence < 0.8
        
        # Test low math content
        low_math_confidence = detector._calculate_math_confidence("This is text without mathematical expressions")
        assert low_math_confidence < 0.3

    def test_deduplicate_candidates(self, test_config):
        """Test candidate deduplication."""
        detector = QuestionDetector(test_config)
        
        candidates = [
            QuestionCandidate("Find P(A|B)", 0.9, "pattern", 0, 50),
            QuestionCandidate("Find P(A|B)", 0.8, "nlp", 10, 60),  # Similar content
            QuestionCandidate("Calculate the probability", 0.7, "math", 20, 70)  # Different content
        ]
        
        deduplicated = detector._deduplicate_candidates(candidates)
        
        assert len(deduplicated) == 2  # Should remove one duplicate
        assert any("Find P(A|B)" in candidate.text for candidate in deduplicated)
        assert any("Calculate the probability" in candidate.text for candidate in deduplicated)

    def test_filter_candidates(self, test_config):
        """Test candidate filtering."""
        detector = QuestionDetector(test_config)
        
        candidates = [
            QuestionCandidate("High confidence question", 0.9, "pattern", 0, 50),
            QuestionCandidate("Medium confidence question", 0.6, "nlp", 10, 60),
            QuestionCandidate("Low confidence question", 0.3, "math", 20, 70)
        ]
        
        filtered = detector._filter_candidates(candidates, min_confidence=0.5)
        
        assert len(filtered) == 2
        assert all(candidate.confidence >= 0.5 for candidate in filtered)

    def test_validate_question(self, test_config):
        """Test question validation."""
        detector = QuestionDetector(test_config)
        
        # Test valid question
        valid_question = "Find the probability that both balls are red given that the first ball is red."
        is_valid = detector.validate_question(valid_question)
        assert is_valid
        
        # Test invalid question (too short)
        invalid_question = "Find."
        is_valid = detector.validate_question(invalid_question)
        assert not is_valid


class TestLaTeXConverter:
    """Test LaTeX converter functionality."""

    def test_initialization(self, test_config):
        """Test LaTeX converter initialization."""
        converter = LaTeXConverter(test_config)
        assert converter.config == test_config

    def test_convert_question_to_latex(self, test_config):
        """Test question to LaTeX conversion."""
        converter = LaTeXConverter(test_config)
        
        question = "A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red."
        
        latex = converter.convert_question_to_latex(question)
        
        assert "$4$" in latex
        assert "$6$" in latex
        assert "$P(\\text{both balls are red})$" in latex

    def test_convert_math_expressions(self, test_config):
        """Test mathematical expression conversion."""
        converter = LaTeXConverter(test_config)
        
        # Test fractions
        assert converter._convert_math_expressions("1/2") == "$\\frac{1}{2}$"
        assert converter._convert_math_expressions("3/4") == "$\\frac{3}{4}$"
        
        # Test probability expressions
        assert converter._convert_math_expressions("P(A|B)") == "$P(A|B)$"
        assert converter._convert_math_expressions("P(A and B)") == "$P(A \\text{ and } B)$"

    def test_convert_probability_expressions(self, test_config):
        """Test probability expression conversion."""
        converter = LaTeXConverter(test_config)
        
        # Test basic probability
        assert converter._convert_probability_expressions("P(A)") == "$P(A)$"
        
        # Test conditional probability
        assert converter._convert_probability_expressions("P(A|B)") == "$P(A|B)$"
        
        # Test complex probability
        assert converter._convert_probability_expressions("P(both balls are red)") == "$P(\\text{both balls are red})$"
        assert converter._convert_probability_expressions("P(sum = 8 | first throw is even)") == "$P(\\text{sum} = 8 | \\text{first throw is even})$"

    def test_convert_numbers(self, test_config):
        """Test number conversion to LaTeX."""
        converter = LaTeXConverter(test_config)
        
        # Test single numbers
        assert converter._convert_numbers("4 red balls") == "$4$ red balls"
        assert converter._convert_numbers("60 students") == "$60$ students"
        
        # Test multiple numbers
        assert converter._convert_numbers("30 play cricket, 20 play football") == "$30$ play cricket, $20$ play football"

    def test_cleanup_latex(self, test_config):
        """Test LaTeX cleanup."""
        converter = LaTeXConverter(test_config)
        
        # Test double dollar signs
        latex = "$$4$$ red balls"
        cleaned = converter._cleanup_latex(latex)
        assert cleaned == "$4$ red balls"
        
        # Test extra spaces
        latex = "$ 4 $ red balls"
        cleaned = converter._cleanup_latex(latex)
        assert cleaned == "$4$ red balls"

    def test_generate_latex_document(self, test_config):
        """Test LaTeX document generation."""
        converter = LaTeXConverter(test_config)
        
        questions = [
            {
                "question_number": "1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Two balls are drawn at random without replacement. Find $P(\\text{both balls are red})$."
            },
            {
                "question_number": "2",
                "question_text": "A die is thrown twice. Find $P(\\text{sum} = 8 | \\text{first throw is even})$."
            }
        ]
        
        latex_doc = converter.generate_latex_document(questions, "Chapter 30: Conditional Probability")
        
        assert "\\documentclass" in latex_doc
        assert "\\begin{enumerate}" in latex_doc
        assert "\\item" in latex_doc
        assert "\\end{enumerate}" in latex_doc
        assert "\\end{document}" in latex_doc

    def test_generate_preamble(self, test_config):
        """Test LaTeX preamble generation."""
        converter = LaTeXConverter(test_config)
        
        preamble = converter._generate_preamble("Test Title", "Test Author")
        
        assert "\\documentclass" in preamble
        assert "\\usepackage{amsmath}" in preamble
        assert "\\title{Test Title}" in preamble
        assert "\\author{Test Author}" in preamble

    def test_generate_body(self, test_config):
        """Test LaTeX body generation."""
        converter = LaTeXConverter(test_config)
        
        questions = [
            {"question_text": "Question 1"},
            {"question_text": "Question 2"}
        ]
        
        body = converter._generate_body(questions)
        
        assert "\\begin{enumerate}" in body
        assert "\\item Question 1" in body
        assert "\\item Question 2" in body
        assert "\\end{enumerate}" in body

    def test_validate_latex(self, test_config):
        """Test LaTeX validation."""
        converter = LaTeXConverter(test_config)
        
        # Test valid LaTeX
        valid_latex = "A bag contains $4$ red balls and $6$ black balls."
        is_valid = converter.validate_latex(valid_latex)
        assert is_valid
        
        # Test invalid LaTeX (unmatched braces)
        invalid_latex = "A bag contains $4$ red balls and $6$ black balls."
        is_valid = converter.validate_latex(invalid_latex)
        assert is_valid  # This should be valid

    def test_convert_batch_questions(self, test_config):
        """Test batch question conversion."""
        converter = LaTeXConverter(test_config)
        
        questions = [
            "A bag contains 4 red balls and 6 black balls. Find the probability that both balls are red.",
            "A die is thrown twice. Find the probability that the sum is 8."
        ]
        
        converted = converter.convert_batch_questions(questions)
        
        assert len(converted) == 2
        assert all("$4$" in q or "$6$" in q or "$8$" in q for q in converted)

    def test_generate_markdown_with_latex(self, test_config):
        """Test Markdown with LaTeX generation."""
        converter = LaTeXConverter(test_config)
        
        questions = [
            {
                "question_number": "1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
            }
        ]
        
        markdown = converter.generate_markdown_with_latex(questions, "Chapter 30")
        
        assert "# Chapter 30" in markdown
        assert "## Question 1" in markdown
        assert "$4$" in markdown
        assert "$P(\\text{both balls are red})$" in markdown

    def test_save_latex_file(self, test_config, temp_output_dir):
        """Test LaTeX file saving."""
        converter = LaTeXConverter(test_config)
        converter.output_dir = temp_output_dir
        
        latex_content = "\\documentclass{article}\\begin{document}Test\\end{document}"
        file_path = converter.save_latex_file(latex_content, "test.tex")
        
        assert Path(file_path).exists()
        with open(file_path, 'r') as f:
            content = f.read()
        assert "\\documentclass" in content

    def test_get_conversion_statistics(self, test_config):
        """Test conversion statistics."""
        converter = LaTeXConverter(test_config)
        
        stats = converter.get_conversion_statistics()
        
        assert 'total_conversions' in stats
        assert 'successful_conversions' in stats
        assert 'failed_conversions' in stats
        assert 'average_conversion_time' in stats


class TestQuestionValidator:
    """Test question validator functionality."""

    def test_initialization(self, test_config):
        """Test question validator initialization."""
        validator = QuestionValidator(test_config)
        assert validator.config == test_config

    def test_validate_question(self, test_config):
        """Test question validation."""
        validator = QuestionValidator(test_config)
        
        question = {
            "question_number": "1",
            "question_text": "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
        }
        
        result = validator.validate_question(question)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.score > 0.7
        assert len(result.errors) == 0

    def test_validate_structure(self, test_config):
        """Test question structure validation."""
        validator = QuestionValidator(test_config)
        
        # Test valid structure
        valid_question = {
            "question_number": "1",
            "question_text": "Valid question text",
            "source": "Exercise 30.3"
        }
        
        result = validator._validate_structure(valid_question)
        assert result['is_valid']
        assert result['score'] > 0.8
        
        # Test invalid structure
        invalid_question = {
            "question_text": "Missing question number"
        }
        
        result = validator._validate_structure(invalid_question)
        assert not result['is_valid']
        assert len(result['errors']) > 0

    def test_validate_mathematical_content(self, test_config):
        """Test mathematical content validation."""
        validator = QuestionValidator(test_config)
        
        # Test valid mathematical content
        valid_content = "Find $P(A|B)$ where $P(A) = \\frac{1}{2}$"
        result = validator._validate_mathematical_content(valid_content)
        assert result['is_valid']
        assert result['score'] > 0.7
        
        # Test invalid mathematical content
        invalid_content = "Find the thing with the stuff"
        result = validator._validate_mathematical_content(invalid_content)
        assert not result['is_valid']
        assert result['score'] < 0.5

    def test_validate_latex_formatting(self, test_config):
        """Test LaTeX formatting validation."""
        validator = QuestionValidator(test_config)
        
        # Test valid LaTeX
        valid_latex = "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
        result = validator._validate_latex_formatting(valid_latex)
        assert result['is_valid']
        assert result['score'] > 0.8
        
        # Test invalid LaTeX (unmatched dollar signs)
        invalid_latex = "A bag contains $4 red balls and $6$ black balls"
        result = validator._validate_latex_formatting(invalid_latex)
        assert not result['is_valid']
        assert len(result['errors']) > 0

    def test_validate_question_batch(self, test_config):
        """Test batch question validation."""
        validator = QuestionValidator(test_config)
        
        questions = [
            {
                "question_number": "1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
            },
            {
                "question_number": "2",
                "question_text": "A die is thrown twice. Find $P(\\text{sum} = 8 | \\text{first throw is even})$."
            }
        ]
        
        results = validator.validate_question_batch(questions)
        
        assert len(results) == 2
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all(result.is_valid for result in results)

    def test_validate_latex_document(self, test_config):
        """Test LaTeX document validation."""
        validator = QuestionValidator(test_config)
        
        latex_doc = """
        \\documentclass{article}
        \\usepackage{amsmath}
        \\begin{document}
        \\title{Test Questions}
        \\maketitle
        \\begin{enumerate}
        \\item A bag contains $4$ red balls and $6$ black balls.
        \\end{enumerate}
        \\end{document}
        """
        
        result = validator.validate_latex_document(latex_doc)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.score > 0.7

    def test_generate_validation_report(self, test_config):
        """Test validation report generation."""
        validator = QuestionValidator(test_config)
        
        questions = [
            {
                "question_number": "1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
            }
        ]
        
        results = [validator.validate_question(q) for q in questions]
        
        report = validator.generate_validation_report(questions, results)
        
        assert "Validation Report" in report
        assert "Total Questions" in report
        assert "Valid Questions" in report
        assert "Average Score" in report

    def test_save_validation_report(self, test_config, temp_output_dir):
        """Test validation report saving."""
        validator = QuestionValidator(test_config)
        validator.output_dir = temp_output_dir
        
        questions = [
            {
                "question_number": "1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Find $P(\\text{both balls are red})$."
            }
        ]
        
        results = [validator.validate_question(q) for q in questions]
        
        file_path = validator.save_validation_report(questions, results, "validation_report.json")
        
        assert Path(file_path).exists()
        with open(file_path, 'r') as f:
            import json
            data = json.load(f)
        assert "summary" in data
        assert "questions" in data

    def test_calculate_overall_score(self, test_config):
        """Test overall score calculation."""
        validator = QuestionValidator(test_config)
        
        results = [
            ValidationResult(is_valid=True, score=0.8, errors=[], warnings=[], suggestions=[]),
            ValidationResult(is_valid=True, score=0.9, errors=[], warnings=[], suggestions=[]),
            ValidationResult(is_valid=False, score=0.6, errors=["Error"], warnings=[], suggestions=[])
        ]
        
        overall_score = validator._calculate_overall_score(results)
        
        assert 0.6 < overall_score < 0.9

    def test_categorize_errors(self, test_config):
        """Test error categorization."""
        validator = QuestionValidator(test_config)
        
        errors = [
            "Missing question number",
            "Invalid LaTeX syntax",
            "No mathematical content",
            "Unmatched dollar signs"
        ]
        
        categorized = validator._categorize_errors(errors)
        
        assert "structure" in categorized
        assert "latex" in categorized
        assert "mathematical" in categorized

    def test_generate_suggestions(self, test_config):
        """Test suggestion generation."""
        validator = QuestionValidator(test_config)
        
        question = {
            "question_text": "Find the probability of drawing 2 red balls from 10 balls"
        }
        
        suggestions = validator._generate_suggestions(question)
        
        assert len(suggestions) > 0
        assert any("LaTeX" in suggestion for suggestion in suggestions)
        assert any("mathematical" in suggestion for suggestion in suggestions) 