"""
Content parser for RD Sharma Question Extractor.

This module handles text structure analysis, content classification,
and mathematical expression detection.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import spacy
from collections import Counter

from ..utils.logger import get_logger
from ..utils.exceptions import DocumentProcessingError
from ..config import config

logger = get_logger(__name__)


@dataclass
class ContentStructure:
    """Content structure analysis result."""
    paragraphs: int
    sentences: int
    words: int
    math_expressions: int
    content_types: List[str]
    question_indicators: List[str]
    mathematical_density: float
    complexity_score: float


@dataclass
class ContentRelationship:
    """Content relationship information."""
    source_type: str
    target_type: str
    relationship: str
    confidence: float


class ContentParser:
    """Handles content parsing and analysis."""

    def __init__(self, config):
        """Initialize content parser."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize NLP model
        self.nlp = None
        self._initialize_nlp()
        
        # Content type patterns
        self.content_patterns = {
            'illustration': [
                r'Illustration\s+\d+',
                r'Example\s+\d+',
                r'Worked\s+Example\s+\d+'
            ],
            'exercise': [
                r'Exercise\s+\d+\.\d+',
                r'Practice\s+Exercise\s+\d+',
                r'Problem\s+\d+'
            ],
            'theory': [
                r'Theory',
                r'Definition',
                r'Theorem',
                r'Lemma',
                r'Corollary',
                r'Proof',
                r'Note:',
                r'Remark:'
            ],
            'solution': [
                r'Solution',
                r'Answer',
                r'Step\s+\d+',
                r'Therefore',
                r'Hence'
            ],
            'question': [
                r'Find\s+the',
                r'Calculate\s+the',
                r'Determine\s+the',
                r'Prove\s+that',
                r'Show\s+that',
                r'If\s+.+\s+then\s+find'
            ]
        }
        
        # Mathematical expression patterns
        self.math_patterns = {
            'fractions': r'\d+/\d+',
            'probabilities': r'P\([^)]*\)',
            'conditional_prob': r'P\([^|]*\|[^)]*\)',
            'sets': r'[A-Z]\s*[∩∪⊂⊃∈∉]',
            'equations': r'[a-zA-Z]\s*=\s*[^=]+',
            'inequalities': r'[a-zA-Z]\s*[<>≤≥]\s*[^<>≤≥]+',
            'summations': r'∑[^∑]*',
            'integrals': r'∫[^∫]*',
            'roots': r'√[^√]*',
            'subscripts': r'[a-zA-Z]_[a-zA-Z0-9]',
            'superscripts': r'[a-zA-Z]\^[a-zA-Z0-9]'
        }
        
        # Question indicators
        self.question_indicators = [
            'find', 'calculate', 'determine', 'compute', 'evaluate',
            'prove', 'show', 'verify', 'demonstrate', 'establish',
            'what is', 'how many', 'how much', 'which', 'when',
            'where', 'why', 'how'
        ]

    def _initialize_nlp(self):
        """Initialize NLP model."""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("NLP model loaded successfully")
        except OSError:
            # Fallback to basic model
            self.nlp = spacy.blank("en")
            self.logger.warning("Using basic NLP model (install en_core_web_sm for better results)")

    def classify_content_type(self, text: str) -> str:
        """
        Classify content type based on text patterns.
        
        Args:
            text: Text to classify
            
        Returns:
            Content type classification
        """
        text_lower = text.lower()
        
        # Check for specific patterns
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return content_type
        
        # Check for question indicators
        if any(indicator in text_lower for indicator in self.question_indicators):
            return 'question'
        
        # Check for mathematical content
        if self._has_mathematical_content(text):
            return 'mathematical'
        
        # Default classification
        return 'content'

    def detect_mathematical_expressions(self, text: str) -> List[str]:
        """
        Detect mathematical expressions in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of mathematical expressions found
        """
        expressions = []
        
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        # Remove duplicates while preserving order
        unique_expressions = []
        for expr in expressions:
            if expr not in unique_expressions:
                unique_expressions.append(expr)
        
        return unique_expressions

    def identify_question_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify question boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of question boundary information
        """
        boundaries = []
        
        # Split into sentences
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Check if sentence contains question indicators
            if self._is_question_sentence(sentence):
                boundaries.append({
                    'type': 'question',
                    'sentence_index': i,
                    'text': sentence,
                    'confidence': self._calculate_question_confidence(sentence)
                })
        
        return boundaries

    def extract_question_numbering(self, text: str) -> List[str]:
        """
        Extract question numbering from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of question numbers found
        """
        numbering_patterns = [
            r'(\d+)\.',  # 1.
            r'\((\d+)\)',  # (1)
            r'(\d+)\)',  # 1)
            r'Question\s+(\d+)',  # Question 1
            r'Problem\s+(\d+)',  # Problem 1
            r'Illustration\s+(\d+)',  # Illustration 1
            r'Example\s+(\d+)',  # Example 1
        ]
        
        numbers = []
        for pattern in numbering_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))  # Remove duplicates

    def analyze_content_structure(self, text: str) -> ContentStructure:
        """
        Analyze content structure and characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            ContentStructure object with analysis results
        """
        # Basic text statistics
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        
        if self.nlp:
            doc = self.nlp(text)
            sentences = len(list(doc.sents))
            words = len([token for token in doc if not token.is_space])
        else:
            # Fallback calculations
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
        
        # Mathematical content analysis
        math_expressions = self.detect_mathematical_expressions(text)
        mathematical_density = len(math_expressions) / max(1, words)
        
        # Content type analysis
        content_types = []
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    content_types.append(content_type)
                    break
        
        # Question indicators
        question_indicators = []
        for indicator in self.question_indicators:
            if indicator in text.lower():
                question_indicators.append(indicator)
        
        # Complexity score
        complexity_score = self._calculate_complexity_score(text)
        
        return ContentStructure(
            paragraphs=paragraphs,
            sentences=sentences,
            words=words,
            math_expressions=len(math_expressions),
            content_types=list(set(content_types)),
            question_indicators=list(set(question_indicators)),
            mathematical_density=mathematical_density,
            complexity_score=complexity_score
        )

    def extract_content_relationships(self, text: str) -> List[ContentRelationship]:
        """
        Extract relationships between different content types.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of content relationships
        """
        relationships = []
        
        # Analyze text structure
        structure = self.analyze_content_structure(text)
        
        # Theory to Example relationships
        if 'theory' in structure.content_types and 'illustration' in structure.content_types:
            relationships.append(ContentRelationship(
                source_type='theory',
                target_type='illustration',
                relationship='demonstrates',
                confidence=0.8
            ))
        
        # Example to Exercise relationships
        if 'illustration' in structure.content_types and 'exercise' in structure.content_types:
            relationships.append(ContentRelationship(
                source_type='illustration',
                target_type='exercise',
                relationship='prepares_for',
                confidence=0.7
            ))
        
        # Theory to Exercise relationships
        if 'theory' in structure.content_types and 'exercise' in structure.content_types:
            relationships.append(ContentRelationship(
                source_type='theory',
                target_type='exercise',
                relationship='applies',
                confidence=0.9
            ))
        
        return relationships

    def validate_content_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate content quality and completeness.
        
        Args:
            text: Text to validate
            
        Returns:
            Quality validation result
        """
        structure = self.analyze_content_structure(text)
        
        # Quality metrics
        score = 100.0
        errors = []
        warnings = []
        
        # Check minimum length
        if structure.words < 10:
            score -= 20
            errors.append("Content too short")
        
        # Check for mathematical content
        if structure.math_expressions == 0:
            score -= 15
            warnings.append("No mathematical expressions detected")
        
        # Check for question indicators
        if not structure.question_indicators:
            score -= 10
            warnings.append("No question indicators found")
        
        # Check sentence structure
        if structure.sentences < 2:
            score -= 10
            warnings.append("Very few sentences")
        
        # Check mathematical density
        if structure.mathematical_density < 0.01:
            score -= 10
            warnings.append("Low mathematical content density")
        
        return {
            'score': max(0, score),
            'errors': errors,
            'warnings': warnings,
            'structure': structure
        }

    def clean_and_normalize_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\+\-\*\/\=\<\>]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract nouns, verbs, and adjectives
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                    not token.is_stop and 
                    len(token.text) > 2):
                    keywords.append(token.text.lower())
            
            # Count frequencies
            keyword_counts = Counter(keywords)
            
            # Return most common keywords
            return [keyword for keyword, count in keyword_counts.most_common(max_keywords)]
        else:
            # Fallback: extract words that appear multiple times
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = {word: count for word, count in word_counts.items() 
                            if word not in stop_words and len(word) > 2}
            
            # Return most common words
            return [word for word, count in sorted(filtered_words.items(), 
                                                 key=lambda x: x[1], reverse=True)[:max_keywords]]

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'hi')
        """
        # Simple language detection based on character sets
        # This is a basic implementation - consider using langdetect library for better results
        
        # Check for Hindi characters
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        if len(hindi_chars) > len(text) * 0.1:  # More than 10% Hindi characters
            return 'hi'
        
        # Check for English (default)
        english_chars = re.findall(r'[a-zA-Z]', text)
        if len(english_chars) > len(text) * 0.3:  # More than 30% English characters
            return 'en'
        
        # Default to English
        return 'en'

    def _has_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        return len(self.detect_mathematical_expressions(text)) > 0

    def _is_question_sentence(self, sentence: str) -> bool:
        """Check if a sentence is a question."""
        sentence_lower = sentence.lower()
        
        # Check for question indicators
        if any(indicator in sentence_lower for indicator in self.question_indicators):
            return True
        
        # Check for question mark
        if '?' in sentence:
            return True
        
        # Check for imperative forms
        imperative_patterns = [
            r'^Find\s+',
            r'^Calculate\s+',
            r'^Determine\s+',
            r'^Prove\s+',
            r'^Show\s+'
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False

    def _calculate_question_confidence(self, sentence: str) -> float:
        """Calculate confidence that a sentence is a question."""
        confidence = 0.0
        
        # Question mark
        if '?' in sentence:
            confidence += 0.4
        
        # Question indicators
        for indicator in self.question_indicators:
            if indicator in sentence.lower():
                confidence += 0.3
                break
        
        # Imperative patterns
        imperative_patterns = [
            r'^Find\s+',
            r'^Calculate\s+',
            r'^Determine\s+',
            r'^Prove\s+',
            r'^Show\s+'
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence += 0.3
                break
        
        return min(1.0, confidence)

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score."""
        if not text:
            return 0.0
        
        # Basic complexity factors
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        math_expressions = len(self.detect_mathematical_expressions(text))
        
        # Average words per sentence
        avg_words_per_sentence = words / max(1, sentences)
        
        # Mathematical density
        math_density = math_expressions / max(1, words)
        
        # Complexity score (0-1)
        complexity = min(1.0, (
            (avg_words_per_sentence / 20.0) * 0.4 +  # Sentence length factor
            (math_density * 10.0) * 0.4 +           # Mathematical content factor
            (len(text) / 1000.0) * 0.2              # Overall length factor
        ))
        
        return complexity 