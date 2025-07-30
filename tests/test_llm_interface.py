"""
Tests for LLM interface module.

This module tests the Groq client integration, prompt engineering,
and response parsing functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
sys.path.append('../src')

from llm_interface.groq_client import GroqClient
from utils.exceptions import LLMInterfaceError


class TestGroqClient:
    """Test cases for GroqClient class."""
    
    @pytest.fixture
    def mock_groq_client(self):
        """Create a mock Groq client for testing."""
        with patch('llm_interface.groq_client.Groq') as mock_groq:
            client = GroqClient(api_key="test_api_key")
            yield client
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return """Chapter 30: Probability

30.3 Conditional Probability

Illustration 1: A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red.

Exercise 30.3

1. A die is thrown twice. Find the probability that the sum of numbers appearing is 8, given that the first throw shows an even number.

2. In a class of 60 students, 30 play cricket, 20 play football and 10 play both games. A student is selected at random. Find the probability that:
   (i) He plays cricket given that he plays football
   (ii) He plays exactly one game"""
    
    @pytest.fixture
    def expected_questions(self):
        """Expected questions from sample content."""
        return [
            {
                "question_number": "Illustration 1",
                "question_text": "A bag contains $4$ red balls and $6$ black balls. Two balls are drawn at random without replacement. Find $P(\\text{both balls are red})$.",
                "source": "Illustration"
            },
            {
                "question_number": "1",
                "question_text": "A die is thrown twice. Find $P(\\text{sum} = 8 | \\text{first throw is even})$.",
                "source": "Exercise 30.3"
            },
            {
                "question_number": "2",
                "question_text": "In a class of $60$ students, $30$ play cricket, $20$ play football and $10$ play both games. A student is selected at random. Find: (i) $P(\\text{cricket} | \\text{football})$ (ii) $P(\\text{exactly one game})$",
                "source": "Exercise 30.3"
            }
        ]
    
    def test_groq_client_initialization(self, mock_groq_client):
        """Test GroqClient initialization."""
        assert mock_groq_client.api_key == "test_api_key"
        assert mock_groq_client.model == "meta-llama-4-maverick-17b"
        assert mock_groq_client.temperature == 0.1
        assert mock_groq_client.max_tokens == 4000
    
    def test_build_extraction_prompt(self, mock_groq_client, sample_content):
        """Test prompt building functionality."""
        prompt = mock_groq_client._build_extraction_prompt(sample_content, 30, "30.3")
        
        # Check that prompt contains key elements
        assert "You are an expert mathematical content extractor" in prompt
        assert "Chapter 30: 30.3" in prompt
        assert sample_content in prompt
        assert "REQUIRED OUTPUT FORMAT" in prompt
        assert "LATEX FORMATTING EXAMPLES" in prompt
    
    def test_parse_extraction_response_valid_json(self, mock_groq_client, expected_questions):
        """Test parsing valid JSON response."""
        response = json.dumps(expected_questions)
        parsed_questions = mock_groq_client._parse_extraction_response(response)
        
        assert len(parsed_questions) == 3
        assert parsed_questions[0]["question_number"] == "Illustration 1"
        assert "$4$" in parsed_questions[0]["question_text"]
        assert "$P(" in parsed_questions[1]["question_text"]
    
    def test_parse_extraction_response_invalid_json(self, mock_groq_client):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        with pytest.raises(LLMInterfaceError):
            mock_groq_client._parse_extraction_response(invalid_response)
    
    def test_parse_extraction_response_no_json_array(self, mock_groq_client):
        """Test parsing response without JSON array."""
        response = "No JSON array here"
        parsed_questions = mock_groq_client._parse_extraction_response(response)
        
        assert parsed_questions == []
    
    def test_validate_response_quality(self, mock_groq_client, expected_questions):
        """Test response validation functionality."""
        validation_results = mock_groq_client.validate_response(expected_questions)
        
        assert validation_results["total_questions"] == 3
        assert validation_results["valid_questions"] == 3
        assert validation_results["quality_score"] == 1.0
        assert len(validation_results["latex_errors"]) == 0
        assert len(validation_results["missing_fields"]) == 0
    
    def test_validate_response_with_errors(self, mock_groq_client):
        """Test validation with problematic questions."""
        problematic_questions = [
            {
                "question_number": "1",
                "question_text": "Find the probability of getting 4 heads in 6 tosses",  # No LaTeX
                "source": "Exercise"
            },
            {
                "question_text": "Missing question number",  # Missing field
                "source": "Exercise"
            }
        ]
        
        validation_results = mock_groq_client.validate_response(problematic_questions)
        
        assert validation_results["total_questions"] == 2
        assert validation_results["valid_questions"] == 0
        assert validation_results["quality_score"] == 0.0
        assert len(validation_results["latex_errors"]) > 0
        assert len(validation_results["missing_fields"]) > 0
    
    def test_has_latex_numbers(self, mock_groq_client):
        """Test LaTeX number detection."""
        text_with_latex = "A bag contains $4$ red balls and $6$ black balls"
        text_without_latex = "A bag contains 4 red balls and 6 black balls"
        
        assert mock_groq_client._has_latex_numbers(text_with_latex) == True
        assert mock_groq_client._has_latex_numbers(text_without_latex) == False
    
    def test_has_latex_probability(self, mock_groq_client):
        """Test LaTeX probability detection."""
        text_with_prob = "Find $P(\\text{sum} = 8)$"
        text_without_prob = "Find the probability of sum being 8"
        
        assert mock_groq_client._has_latex_probability(text_with_prob) == True
        assert mock_groq_client._has_latex_probability(text_without_prob) == False
    
    @patch('llm_interface.groq_client.time.sleep')
    def test_rate_limiting(self, mock_sleep, mock_groq_client):
        """Test rate limiting functionality."""
        # First call should not sleep
        mock_groq_client._rate_limit()
        mock_sleep.assert_not_called()
        
        # Second call immediately after should sleep
        mock_groq_client._rate_limit()
        mock_sleep.assert_called_once()
    
    @patch('llm_interface.groq_client.time.sleep')
    def test_make_request_with_retries(self, mock_sleep, mock_groq_client):
        """Test request retry mechanism."""
        # Mock the Groq client to fail twice then succeed
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([{"question_text": "test"}])
        
        mock_groq_client.client.chat.completions.create.side_effect = [
            Exception("API Error"),  # First attempt fails
            Exception("API Error"),  # Second attempt fails
            mock_response  # Third attempt succeeds
        ]
        
        messages = [{"role": "user", "content": "test"}]
        response = mock_groq_client._make_request(messages)
        
        assert response == mock_response
        assert mock_groq_client.client.chat.completions.create.call_count == 3
    
    def test_make_request_max_retries_exceeded(self, mock_groq_client):
        """Test request failure after max retries."""
        mock_groq_client.client.chat.completions.create.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(LLMInterfaceError):
            mock_groq_client._make_request(messages)
    
    def test_format_latex(self, mock_groq_client):
        """Test LaTeX formatting functionality."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Find $P(\\text{event})$"
        
        mock_groq_client._make_request.return_value = mock_response
        
        original_text = "Find the probability of event"
        formatted_text = mock_groq_client.format_latex(original_text)
        
        assert formatted_text == "Find $P(\\text{event})$"
        mock_groq_client._make_request.assert_called_once()
    
    def test_format_latex_failure_fallback(self, mock_groq_client):
        """Test LaTeX formatting fallback on failure."""
        mock_groq_client._make_request.side_effect = Exception("API Error")
        
        original_text = "Find the probability of event"
        formatted_text = mock_groq_client.format_latex(original_text)
        
        # Should return original text on failure
        assert formatted_text == original_text


class TestLLMInterfaceIntegration:
    """Integration tests for LLM interface."""
    
    def test_end_to_end_extraction(self):
        """Test complete question extraction pipeline."""
        # This would require a real API key for full testing
        # For now, we'll test the structure
        with patch('llm_interface.groq_client.Groq'):
            client = GroqClient(api_key="test_key")
            
            # Test that the client can be created
            assert client is not None
            assert hasattr(client, 'extract_questions')
            assert hasattr(client, 'validate_response')
    
    def test_error_handling(self):
        """Test error handling in LLM interface."""
        with patch('llm_interface.groq_client.Groq'):
            client = GroqClient(api_key="test_key")
            
            # Test with invalid content
            with pytest.raises(Exception):
                client.extract_questions("", 1, "1.1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 