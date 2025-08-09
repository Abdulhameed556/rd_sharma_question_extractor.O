"""
Question detection for RD Sharma Question Extractor.

This module provides intelligent question detection capabilities for
mathematical content using pattern matching and ML-based classification.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import spacy
from collections import defaultdict

from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError
from src.config import config

logger = get_logger(__name__)


@dataclass
class QuestionCandidate:
    """Represents a detected question candidate."""
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    question_type: str
    source: str
    metadata: Dict[str, Any]


class QuestionDetector:
    """Detects mathematical questions in content using multiple strategies."""
    
    def __init__(self):
        """Initialize question detector with pattern matching and NLP."""
        self.nlp = None
        self._initialize_nlp()
        
        # Question patterns for different types
        self.question_patterns = {
            "illustration": [
                r"Illustration\s+\d+[:\s]*",
                r"Example\s+\d+[:\s]*",
                r"Solved\s+Example\s+\d+[:\s]*"
            ],
            "exercise": [
                r"Exercise\s+\d+[.\d]*[:\s]*",
                r"Question\s+\d+[:\s]*",
                r"Problem\s+\d+[:\s]*",
                r"Practice\s+Question\s+\d+[:\s]*"
            ],
            "mcq": [
                r"Multiple\s+Choice\s+Question\s+\d+[:\s]*",
                r"MCQ\s+\d+[:\s]*",
                r"Choose\s+the\s+correct\s+option[:\s]*"
            ],
            "numerical": [
                r"Numerical\s+Question\s+\d+[:\s]*",
                r"Calculate\s+[^.]*[:\s]*",
                r"Find\s+[^.]*[:\s]*",
                r"Determine\s+[^.]*[:\s]*"
            ]
        }
        
        # Question indicators
        self.question_indicators = [
            "find", "calculate", "determine", "evaluate", "compute",
            "solve", "prove", "show", "verify", "check",
            "what is", "how many", "how much", "which", "where",
            "when", "why", "how", "if", "given that"
        ]
        
        # Non-question patterns (to filter out)
        self.non_question_patterns = [
            r"Solution[:\s]*",
            r"Answer[:\s]*",
            r"Explanation[:\s]*",
            r"Theory[:\s]*",
            r"Definition[:\s]*",
            r"Formula[:\s]*",
            r"Note[:\s]*",
            r"Remark[:\s]*"
        ]
        
        logger.info("Question detector initialized")
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model for text processing."""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found, installing...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model installed and loaded")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}. Using basic text processing.")
                self.nlp = None
    
    def detect_questions(self, content: str, chapter: int, topic: str) -> List[QuestionCandidate]:
        """
        Detect questions in content using multiple detection strategies.
        
        Args:
            content: Text content to analyze
            chapter: Chapter number for context
            topic: Topic identifier for context
            
        Returns:
            List of detected question candidates
        """
        candidates = []
        
        # Strategy 1: Pattern-based detection
        pattern_candidates = self._detect_by_patterns(content, chapter, topic)
        candidates.extend(pattern_candidates)
        
        # Strategy 2: NLP-based detection
        if self.nlp:
            nlp_candidates = self._detect_by_nlp(content, chapter, topic)
            candidates.extend(nlp_candidates)
        
        # Strategy 3: Mathematical expression detection
        math_candidates = self._detect_by_mathematical_content(content, chapter, topic)
        candidates.extend(math_candidates)
        
        # Remove duplicates and filter
        unique_candidates = self._deduplicate_candidates(candidates)
        filtered_candidates = self._filter_candidates(unique_candidates, content)
        
        # Sort by confidence
        filtered_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Detected {len(filtered_candidates)} question candidates")
        return filtered_candidates
    
    def _detect_by_patterns(self, content: str, chapter: int, topic: str) -> List[QuestionCandidate]:
        """Detect questions using pattern matching."""
        candidates = []
        
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Extract question text
                    question_text = self._extract_question_text(content, match.end())
                    
                    if question_text:
                        confidence = self._calculate_pattern_confidence(pattern, question_text)
                        
                        candidate = QuestionCandidate(
                            text=question_text,
                            start_pos=match.start(),
                            end_pos=match.start() + len(question_text),
                            confidence=confidence,
                            question_type=question_type,
                            source=f"pattern_{question_type}",
                            metadata={
                                "pattern": pattern,
                                "match_start": match.start(),
                                "match_end": match.end(),
                                "chapter": chapter,
                                "topic": topic
                            }
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _detect_by_nlp(self, content: str, chapter: int, topic: str) -> List[QuestionCandidate]:
        """Detect questions using NLP analysis."""
        if not self.nlp:
            return []
        
        candidates = []
        doc = self.nlp(content)
        
        # Find sentences that contain question indicators
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check for question indicators
            has_question_indicator = any(
                indicator in sent_text.lower() for indicator in self.question_indicators
            )
            
            # Check for mathematical content
            has_math_content = self._contains_mathematical_content(sent_text)
            
            # Check for question marks or imperative structure
            has_question_structure = (
                "?" in sent_text or
                sent_text.lower().startswith(("find", "calculate", "determine", "solve"))
            )
            
            if (has_question_indicator or has_question_structure) and has_math_content:
                confidence = self._calculate_nlp_confidence(sent_text)
                
                candidate = QuestionCandidate(
                    text=sent_text,
                    start_pos=sent.start_char,
                    end_pos=sent.end_char,
                    confidence=confidence,
                    question_type="nlp_detected",
                    source="nlp_analysis",
                    metadata={
                        "has_question_indicator": has_question_indicator,
                        "has_math_content": has_math_content,
                        "has_question_structure": has_question_structure,
                        "chapter": chapter,
                        "topic": topic
                    }
                )
                candidates.append(candidate)
        
        return candidates
    
    def _detect_by_mathematical_content(self, content: str, chapter: int, topic: str) -> List[QuestionCandidate]:
        """Detect questions based on mathematical content patterns."""
        candidates = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            if not sentence:
                continue
            
            # Check for mathematical patterns
            math_patterns = [
                r'\b\d+\s*[+\-×÷=<>≤≥≠±]\s*\d+\b',  # Arithmetic operations
                r'\b[a-zA-Z]\s*=\s*\d+\b',          # Variable assignments
                r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\([^)]+\)\b',  # Functions
                r'\bP\([^)]+\)\b',                  # Probability expressions
                r'\b\d+\s*[+\-×÷]\s*[a-zA-Z]\b',    # Algebraic expressions
            ]
            
            has_math_pattern = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in math_patterns)
            
            # Check for question-like structure
            has_question_structure = any(
                indicator in sentence.lower() for indicator in self.question_indicators
            )
            
            if has_math_pattern and has_question_structure:
                confidence = self._calculate_math_confidence(sentence)
                
                # Find position in original content
                start_pos = content.find(sentence)
                end_pos = start_pos + len(sentence) if start_pos != -1 else 0
                
                candidate = QuestionCandidate(
                    text=sentence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                    question_type="mathematical",
                    source="math_pattern",
                    metadata={
                        "math_patterns_found": [p for p in math_patterns if re.search(p, sentence, re.IGNORECASE)],
                        "has_question_structure": has_question_structure,
                        "chapter": chapter,
                        "topic": topic
                    }
                )
                candidates.append(candidate)
        
        return candidates
    
    def _extract_question_text(self, content: str, start_pos: int, max_length: int = 500) -> str:
        """Extract question text starting from a given position."""
        if start_pos >= len(content):
            return ""
        
        # Find the end of the question (next pattern or end of reasonable length)
        end_pos = start_pos + max_length
        
        # Look for natural breaks
        for i in range(start_pos, min(end_pos, len(content))):
            if content[i] in '.!?':
                # Check if this is a natural sentence break
                if i + 1 < len(content) and content[i + 1] in ' \n':
                    end_pos = i + 1
                    break
        
        question_text = content[start_pos:end_pos].strip()
        
        # Clean up the text
        question_text = re.sub(r'\s+', ' ', question_text)  # Normalize whitespace
        question_text = question_text.strip()
        
        return question_text
    
    def _contains_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        math_indicators = [
            r'\d+',  # Numbers
            r'[+\-×÷=<>≤≥≠±]',  # Mathematical operators
            r'\b(sin|cos|tan|log|ln|exp|sqrt)\b',  # Functions
            r'\bP\([^)]+\)\b',  # Probability
            r'[a-zA-Z]\s*=\s*\d+',  # Variable assignments
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_indicators)
    
    def _calculate_pattern_confidence(self, pattern: str, question_text: str) -> float:
        """Calculate confidence score for pattern-based detection."""
        confidence = 0.5  # Base confidence
        
        # Pattern strength
        if "Illustration" in pattern or "Example" in pattern:
            confidence += 0.3
        elif "Exercise" in pattern or "Question" in pattern:
            confidence += 0.4
        elif "MCQ" in pattern:
            confidence += 0.2
        
        # Text length factor
        if len(question_text) > 50:
            confidence += 0.1
        if len(question_text) > 100:
            confidence += 0.1
        
        # Mathematical content factor
        if self._contains_mathematical_content(question_text):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_nlp_confidence(self, text: str) -> float:
        """Calculate confidence score for NLP-based detection."""
        confidence = 0.4  # Base confidence
        
        # Question indicator strength
        strong_indicators = ["find", "calculate", "determine", "solve"]
        weak_indicators = ["what", "how", "which", "where", "when", "why"]
        
        text_lower = text.lower()
        
        for indicator in strong_indicators:
            if indicator in text_lower:
                confidence += 0.3
                break
        
        for indicator in weak_indicators:
            if indicator in text_lower:
                confidence += 0.1
                break
        
        # Mathematical content
        if self._contains_mathematical_content(text):
            confidence += 0.2
        
        # Question mark
        if "?" in text:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_math_confidence(self, text: str) -> float:
        """Calculate confidence score for mathematical content detection."""
        confidence = 0.3  # Base confidence
        
        # Mathematical pattern strength
        math_patterns = [
            (r'\b\d+\s*[+\-×÷=<>≤≥≠±]\s*\d+\b', 0.2),  # Arithmetic
            (r'\bP\([^)]+\)\b', 0.3),                   # Probability
            (r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\([^)]+\)\b', 0.2),  # Functions
            (r'\b[a-zA-Z]\s*=\s*\d+\b', 0.1),          # Assignments
        ]
        
        for pattern, score in math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence += score
        
        # Question structure
        if any(indicator in text.lower() for indicator in self.question_indicators):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _deduplicate_candidates(self, candidates: List[QuestionCandidate]) -> List[QuestionCandidate]:
        """Remove duplicate question candidates."""
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Create a key based on text content and position
            key = (candidate.text[:100], candidate.start_pos // 100)  # Approximate position
            
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _filter_candidates(self, candidates: List[QuestionCandidate], content: str) -> List[QuestionCandidate]:
        """Filter candidates based on quality criteria."""
        filtered = []
        
        for candidate in candidates:
            # Skip if too short
            if len(candidate.text) < 20:
                continue
            
            # Skip if matches non-question patterns
            if any(re.search(pattern, candidate.text, re.IGNORECASE) for pattern in self.non_question_patterns):
                continue
            
            # Skip if confidence is too low
            if candidate.confidence < 0.3:
                continue
            
            # Skip if no mathematical content
            if not self._contains_mathematical_content(candidate.text):
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def validate_question(self, candidate: QuestionCandidate) -> Dict[str, Any]:
        """Validate a question candidate and provide quality metrics."""
        validation = {
            "is_valid": True,
            "confidence": candidate.confidence,
            "issues": [],
            "suggestions": []
        }
        
        # Check text length
        if len(candidate.text) < 30:
            validation["issues"].append("Question text too short")
            validation["is_valid"] = False
        
        # Check for mathematical content
        if not self._contains_mathematical_content(candidate.text):
            validation["issues"].append("No mathematical content detected")
            validation["is_valid"] = False
        
        # Check for question structure
        has_question_structure = any(
            indicator in candidate.text.lower() for indicator in self.question_indicators
        )
        
        if not has_question_structure:
            validation["suggestions"].append("Consider adding question indicators")
        
        # Check for completeness
        if not candidate.text.strip().endswith(('.', '?', '!')):
            validation["suggestions"].append("Question may be incomplete")
        
        return validation 