"""
Content chunking for RD Sharma Question Extractor.

This module provides intelligent content chunking capabilities optimized
for mathematical content and textbook structure.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from ..utils.logger import get_logger
from ..utils.exceptions import RAGPipelineError
from ..config import config

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a content chunk with metadata."""
    content: str
    chunk_id: str
    start_pos: int
    end_pos: int
    chunk_type: str
    metadata: Dict[str, Any]


class ContentChunker:
    """Intelligent content chunking for mathematical textbook content."""
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 overlap_size: int = 200,
                 min_chunk_size: int = 100):
        """
        Initialize content chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap_size: Overlap between consecutive chunks
            min_chunk_size: Minimum size of each chunk
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Mathematical content boundaries
        self.math_boundaries = [
            r'\$[^$]*\$',  # Inline math
            r'\\\[[^\]]*\\\]',  # Display math
            r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}',  # LaTeX environments
            r'P\([^)]*\)',  # Probability expressions
            r'\b\d+\s*[+\-×÷=<>≤≥≠±]\s*\d+\b',  # Arithmetic operations
        ]
        
        # Content type patterns
        self.content_patterns = {
            "illustration": r"Illustration\s+\d+[:\s]*",
            "exercise": r"Exercise\s+\d+[.\d]*[:\s]*",
            "question": r"Question\s+\d+[:\s]*",
            "theory": r"Theory[:\s]*|Definition[:\s]*|Formula[:\s]*",
            "solution": r"Solution[:\s]*|Answer[:\s]*",
            "example": r"Example\s+\d+[:\s]*",
        }
        
        logger.info(f"Content chunker initialized with max_size={max_chunk_size}, overlap={overlap_size}")
    
    def chunk_content(self, content: str, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """
        Chunk content into manageable pieces for RAG processing.
        
        Args:
            content: Raw text content
            chapter: Chapter number for context
            topic: Topic identifier for context
            
        Returns:
            List of content chunks with metadata
        """
        try:
            # Preprocess content
            processed_content = self._preprocess_content(content)
            
            # Detect content structure
            structure = self._analyze_content_structure(processed_content)
            
            # Create chunks based on structure
            if structure["has_clear_boundaries"]:
                chunks = self._chunk_by_structure(processed_content, structure, chapter, topic)
            else:
                chunks = self._chunk_by_size(processed_content, chapter, topic)
            
            # Post-process chunks
            processed_chunks = self._post_process_chunks(chunks)
            
            logger.info(f"Created {len(processed_chunks)} chunks from {len(content)} characters")
            return processed_chunks
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to chunk content: {str(e)}",
                component="chunker",
                context={"content_length": len(content)}
            )
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for better chunking."""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Clean up mathematical expressions
        content = self._clean_math_expressions(content)
        
        return content.strip()
    
    def _clean_math_expressions(self, content: str) -> str:
        """Clean and normalize mathematical expressions."""
        # Fix common OCR issues in math
        replacements = {
            r'\b(\d+)\s*[xX]\s*(\d+)\b': r'\1 × \2',  # Fix multiplication
            r'\b(\d+)\s*/\s*(\d+)\b': r'\1 ÷ \2',     # Fix division
            r'\b(\d+)\s*=\s*(\d+)\b': r'\1 = \2',     # Fix equality
        }
        
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure to determine chunking strategy."""
        structure = {
            "has_clear_boundaries": False,
            "content_types": [],
            "math_density": 0.0,
            "paragraph_count": 0,
            "sentence_count": 0,
        }
        
        # Count paragraphs and sentences
        paragraphs = content.split('\n\n')
        structure["paragraph_count"] = len([p for p in paragraphs if p.strip()])
        
        sentences = re.split(r'[.!?]+', content)
        structure["sentence_count"] = len([s for s in sentences if s.strip()])
        
        # Detect content types
        for content_type, pattern in self.content_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                structure["content_types"].append(content_type)
        
        # Calculate math density
        math_matches = 0
        for pattern in self.math_boundaries:
            math_matches += len(re.findall(pattern, content))
        
        structure["math_density"] = math_matches / max(len(content.split()), 1)
        
        # Determine if content has clear boundaries
        structure["has_clear_boundaries"] = (
            len(structure["content_types"]) > 0 or
            structure["paragraph_count"] > 5
        )
        
        return structure
    
    def _chunk_by_structure(self, content: str, structure: Dict[str, Any], chapter: int, topic: str) -> List[Chunk]:
        """Chunk content based on structural boundaries."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        current_start = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk, current_start, current_start + len(current_chunk),
                    chapter, topic, structure
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap_size)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + paragraph
                current_start = current_start + overlap_start
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = content.find(paragraph)
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, current_start, current_start + len(current_chunk),
                chapter, topic, structure
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_size(self, content: str, chapter: int, topic: str) -> List[Chunk]:
        """Chunk content by size when no clear structure is detected."""
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            # Determine chunk end position
            end_pos = min(start_pos + self.max_chunk_size, len(content))
            
            # Try to break at sentence boundary
            if end_pos < len(content):
                # Look for sentence boundary
                for i in range(end_pos, max(start_pos + self.min_chunk_size, end_pos - 100), -1):
                    if content[i] in '.!?':
                        end_pos = i + 1
                        break
            
            # Extract chunk content
            chunk_content = content[start_pos:end_pos].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_content, start_pos, end_pos,
                    chapter, topic, {"content_types": ["general"]}
                )
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_pos = max(start_pos + 1, end_pos - self.overlap_size)
        
        return chunks
    
    def _create_chunk(self, content: str, start_pos: int, end_pos: int, 
                     chapter: int, topic: str, structure: Dict[str, Any]) -> Chunk:
        """Create a chunk with metadata."""
        # Generate chunk ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"chunk_{chapter}_{topic.replace('.', '_')}_{content_hash}"
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(content, structure)
        
        # Create metadata
        metadata = {
            "chapter": chapter,
            "topic": topic,
            "content_length": len(content),
            "math_density": structure.get("math_density", 0.0),
            "content_types": structure.get("content_types", []),
            "start_pos": start_pos,
            "end_pos": end_pos,
        }
        
        return Chunk(
            content=content,
            chunk_id=chunk_id,
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_type=chunk_type,
            metadata=metadata
        )
    
    def _determine_chunk_type(self, content: str, structure: Dict[str, Any]) -> str:
        """Determine the type of content in a chunk."""
        content_lower = content.lower()
        
        # Check for specific content types
        if any(re.search(self.content_patterns["illustration"], content, re.IGNORECASE)):
            return "illustration"
        elif any(re.search(self.content_patterns["exercise"], content, re.IGNORECASE)):
            return "exercise"
        elif any(re.search(self.content_patterns["question"], content, re.IGNORECASE)):
            return "question"
        elif any(re.search(self.content_patterns["theory"], content, re.IGNORECASE)):
            return "theory"
        elif any(re.search(self.content_patterns["solution"], content, re.IGNORECASE)):
            return "solution"
        elif any(re.search(self.content_patterns["example"], content, re.IGNORECASE)):
            return "example"
        
        # Check for mathematical content
        if structure.get("math_density", 0) > 0.1:
            return "mathematical"
        
        # Default type
        return "general"
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Post-process chunks and convert to dictionary format."""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if len(chunk.content) < self.min_chunk_size:
                continue
            
            # Convert to dictionary
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
            }
            
            processed_chunks.append(chunk_dict)
        
        return processed_chunks
    
    def merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0].copy()
        
        for next_chunk in chunks[1:]:
            # Check if we should merge
            combined_size = len(current_chunk["content"]) + len(next_chunk["content"])
            
            if combined_size <= self.max_chunk_size:
                # Merge chunks
                current_chunk["content"] += "\n\n" + next_chunk["content"]
                current_chunk["end_pos"] = next_chunk["end_pos"]
                current_chunk["metadata"]["content_length"] = len(current_chunk["content"])
                
                # Merge content types
                current_types = set(current_chunk["metadata"].get("content_types", []))
                next_types = set(next_chunk["metadata"].get("content_types", []))
                current_chunk["metadata"]["content_types"] = list(current_types.union(next_types))
            else:
                # Add current chunk and start new one
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk.copy()
        
        # Add final chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def split_large_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split chunks that are too large."""
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk["content"]) <= self.max_chunk_size:
                split_chunks.append(chunk)
            else:
                # Split large chunk
                sub_chunks = self._split_chunk(chunk)
                split_chunks.extend(sub_chunks)
        
        return split_chunks
    
    def _split_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a single large chunk."""
        content = chunk["content"]
        sub_chunks = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', content)
        current_chunk = ""
        current_start = chunk["start_pos"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create sub-chunk
                sub_chunk = chunk.copy()
                sub_chunk["content"] = current_chunk
                sub_chunk["end_pos"] = current_start + len(current_chunk)
                sub_chunk["metadata"]["content_length"] = len(current_chunk)
                sub_chunks.append(sub_chunk)
                
                # Start new sub-chunk
                current_chunk = sentence
                current_start = chunk["start_pos"] + content.find(sentence)
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final sub-chunk
        if current_chunk:
            sub_chunk = chunk.copy()
            sub_chunk["content"] = current_chunk
            sub_chunk["end_pos"] = current_start + len(current_chunk)
            sub_chunk["metadata"]["content_length"] = len(current_chunk)
            sub_chunks.append(sub_chunk)
        
        return sub_chunks 