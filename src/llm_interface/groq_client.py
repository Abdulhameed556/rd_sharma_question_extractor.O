"""
Groq API client for RD Sharma Question Extractor.

This module provides integration with Groq's Meta-Llama-4-Maverick-17B model
for question extraction and LaTeX formatting with robust error handling.
"""

import time
import json
from typing import Dict, Any, List, Optional, Union
from groq import Groq
from groq.types.chat import ChatCompletion

from src.utils.logger import get_logger, log_exception
from src.utils.exceptions import LLMInterfaceError
from src.config import config

logger = get_logger(__name__)


class GroqClient:
    """Groq API client with Llama-4-Maverick-17B integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (uses config if not provided)
        """
        self.api_key = api_key or config.groq_api_key
        self.model = config.groq_model
        self.client = Groq(api_key=self.api_key)
        
        # Model parameters
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        
        # Rate limiting
        self.requests_per_minute = 60  # Groq's default limit
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.requests_per_minute
        
        # Retry configuration
        self.max_retries = config.retry_attempts
        self.retry_delay = 1.0  # seconds
        
        logger.info(f"Initialized Groq client with model: {self.model}")
    
    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> ChatCompletion:
        """
        Make a request to Groq API with error handling and retries.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the API call
        
        Returns:
            ChatCompletion response from Groq
            
        Raises:
            LLMInterfaceError: If the request fails after retries
        """
        # Set default parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        
        # Apply rate limiting
        self._rate_limit()
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making Groq API request (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = self.client.chat.completions.create(**params)
                
                logger.debug(f"Groq API request successful: {len(response.choices)} choices")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Groq API request failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Groq API request failed after {self.max_retries + 1} attempts")
        
        # If we get here, all retries failed
        raise LLMInterfaceError(
            message=f"Failed to get response from Groq API after {self.max_retries + 1} attempts",
            model=self.model,
            api_response={"error": str(last_exception) if last_exception else "Unknown error"}
        )
    
    def extract_questions(self, content: str, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """
        Extract questions from content using the enhanced prompt.
        
        Args:
            content: Text content to extract questions from
            chapter: Chapter number
            topic: Topic identifier
            
        Returns:
            List of extracted questions with LaTeX formatting
        """
        prompt = self._build_extraction_prompt(content, chapter, topic)
        
        try:
            response = self._make_request([{"role": "user", "content": prompt}])
            
            if not response.choices:
                raise LLMInterfaceError(
                    message="No response choices received from Groq API",
                    model=self.model
                )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {content[:200]}...")
            
            # Parse the response
            questions = self._parse_extraction_response(content)
            
            logger.info(f"Successfully extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            log_exception(logger, e, {"operation": "extract_questions", "chapter": chapter, "topic": topic})
            raise
    
    def _build_extraction_prompt(self, content: str, chapter: int, topic: str) -> str:
        """
        Build the enhanced extraction prompt based on the user's proven prompt.
        
        Args:
            content: Text content to extract questions from
            chapter: Chapter number
            topic: Topic identifier
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert mathematical content extractor specializing in LaTeX formatting for academic publications.
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

CONTENT TO PROCESS:
Chapter {chapter}: {topic}

{content}

REQUIRED OUTPUT FORMAT:
```json
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
```

LATEX FORMATTING EXAMPLES - FOLLOW THESE PATTERNS:
- Numbers: $4$, $6$, $8$, $30$, $52$, $60$
- Basic Probability: $P(\\text{{sum}} = 8)$, $P(\\text{{both aces}})$
- Conditional Probability: $P(A|B)$, $P(\\text{{cricket}}|\\text{{football}})$, $P(\\text{{sum}} = 8 | \\text{{first throw is even}})$
- Complex Events: $P(\\text{{both balls are red}})$, $P(\\text{{exactly one game}})$, $P(\\text{{at least one ace}})$
- Sets: $P(A \\cap B)$, $P(A \\cup B)$
- Text in math: Always use $P(\\text{{descriptive event}})$ format

ADVANCED PROBABILITY EXAMPLES:
- "Find the probability that both balls are red" → "Find $P(\\text{{both balls are red}})$"
- "Find the probability that he plays cricket given that he plays football" → "Find $P(\\text{{cricket}} | \\text{{football}})$"
- "Find the probability that both cards are aces given that at least one is an ace" → "Find $P(\\text{{both aces}} | \\text{{at least one ace}})$"

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
    
    def _parse_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract questions.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            List of parsed questions
        """
        try:
            # Try to extract JSON from the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_str = response[json_start:json_end]
            questions = json.loads(json_str)
            
            # Validate question structure
            validated_questions = []
            for i, question in enumerate(questions):
                if isinstance(question, dict) and "question_text" in question:
                    validated_questions.append(question)
                else:
                    logger.warning(f"Invalid question structure at index {i}: {question}")
            
            return validated_questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response}")
            raise LLMInterfaceError(
                message="Failed to parse JSON response from LLM",
                model=self.model,
                api_response={"response": response, "error": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise LLMInterfaceError(
                message="Unexpected error parsing LLM response",
                model=self.model,
                api_response={"response": response, "error": str(e)}
            )
    
    def format_latex(self, question_text: str) -> str:
        """
        Format a question text to ensure proper LaTeX formatting.
        
        Args:
            question_text: Raw question text
            
        Returns:
            LaTeX-formatted question text
        """
        prompt = f"""Format the following mathematical question text with proper LaTeX formatting:

ORIGINAL TEXT:
{question_text}

REQUIREMENTS:
- All numbers must be in LaTeX: $4$, $6$, $8$
- All mathematical expressions: $P(A|B)$, $\\frac{{1}}{{2}}$
- All probability statements: $P(\\text{{event}})$
- All conditional probabilities: $P(\\text{{A}} | \\text{{B}})$
- All sets: $A \\cap B$, $A \\cup B$

FORMATTED TEXT:"""
        
        try:
            response = self._make_request([{"role": "user", "content": prompt}])
            
            if response.choices:
                formatted_text = response.choices[0].message.content.strip()
                logger.debug(f"LaTeX formatting: {question_text[:50]}... → {formatted_text[:50]}...")
                return formatted_text
            else:
                logger.warning("No response for LaTeX formatting, returning original text")
                return question_text
                
        except Exception as e:
            logger.warning(f"LaTeX formatting failed: {e}, returning original text")
            return question_text
    
    def validate_response(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the extracted questions for quality and completeness.
        
        Args:
            questions: List of extracted questions
            
        Returns:
            Validation results
        """
        validation_results = {
            "total_questions": len(questions),
            "valid_questions": 0,
            "latex_errors": [],
            "missing_fields": [],
            "quality_score": 0.0
        }
        
        for i, question in enumerate(questions):
            # Check required fields
            if not isinstance(question, dict):
                validation_results["missing_fields"].append(f"Question {i}: Not a dictionary")
                continue
            
            required_fields = ["question_number", "question_text", "source"]
            missing = [field for field in required_fields if field not in question]
            
            if missing:
                validation_results["missing_fields"].append(f"Question {i}: Missing {missing}")
                continue
            
            # Check LaTeX formatting
            text = question["question_text"]
            if not self._has_latex_numbers(text):
                validation_results["latex_errors"].append(f"Question {i}: Missing LaTeX numbers")
            
            if not self._has_latex_probability(text):
                validation_results["latex_errors"].append(f"Question {i}: Missing LaTeX probability notation")
            
            validation_results["valid_questions"] += 1
        
        # Calculate quality score
        if validation_results["total_questions"] > 0:
            validation_results["quality_score"] = (
                validation_results["valid_questions"] / validation_results["total_questions"]
            )
        
        return validation_results
    
    def _has_latex_numbers(self, text: str) -> bool:
        """Check if text contains LaTeX-formatted numbers."""
        import re
        # Look for patterns like $4$, $6$, etc.
        return bool(re.search(r'\$\d+\$', text))
    
    def _has_latex_probability(self, text: str) -> bool:
        """Check if text contains LaTeX-formatted probability expressions."""
        import re
        # Look for patterns like $P(...)$, $P(...|...)$, etc.
        return bool(re.search(r'\$P\([^$]*\)\$', text)) 