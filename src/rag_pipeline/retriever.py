"""
RAG retriever for RD Sharma Question Extractor.

This module provides intelligent content retrieval capabilities for
the RAG pipeline with dynamic boundary detection and context management.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.exceptions import RAGPipelineError
from ..config import config

logger = get_logger(__name__)


class RAGRetriever:
    """Intelligent content retriever for RAG pipeline."""
    
    def __init__(self, vector_store, embedding_generator):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        # Retrieval parameters
        self.default_k = 5
        self.similarity_threshold = 0.5
        self.max_context_length = 4000
        
        # Topic boundary detection patterns
        self.topic_patterns = [
            r'Chapter\s+\d+[:\s]*',
            r'\d+\.\d+\s+[A-Z][^.]*[:\s]*',  # e.g., "30.3 Conditional Probability:"
            r'Topic\s+\d+[:\s]*',
            r'Section\s+\d+[:\s]*',
        ]
        
        # Content type weights for retrieval
        self.content_weights = {
            "question": 1.5,
            "exercise": 1.3,
            "illustration": 1.2,
            "example": 1.1,
            "mathematical": 1.0,
            "theory": 0.8,
            "solution": 0.6,
            "general": 0.5,
        }
        
        logger.info("RAG retriever initialized")
    
    def retrieve_for_question_extraction(self, chapter: int, topic: str, 
                                       query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content for question extraction.
        
        Args:
            chapter: Chapter number
            topic: Topic identifier
            query_text: Optional query text for semantic search
            
        Returns:
            List of relevant chunks for question extraction
        """
        try:
            # Strategy 1: Direct chapter/topic retrieval
            direct_results = self._retrieve_by_chapter_topic(chapter, topic)
            
            # Strategy 2: Semantic search if query provided
            semantic_results = []
            if query_text:
                semantic_results = self._retrieve_by_semantic_search(query_text, chapter, topic)
            
            # Strategy 3: Dynamic boundary detection
            boundary_results = self._retrieve_with_boundary_detection(chapter, topic)
            
            # Combine and rank results
            all_results = self._combine_retrieval_results(
                direct_results, semantic_results, boundary_results
            )
            
            # Apply content filtering and ranking
            filtered_results = self._filter_and_rank_results(all_results, chapter, topic)
            
            # Limit context length
            final_results = self._limit_context_length(filtered_results)
            
            logger.info(f"Retrieved {len(final_results)} chunks for chapter {chapter}, topic {topic}")
            return final_results
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to retrieve content: {str(e)}",
                component="retriever",
                context={"chapter": chapter, "topic": topic}
            )
    
    def _retrieve_by_chapter_topic(self, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Retrieve chunks by chapter and topic metadata."""
        try:
            chunks = self.vector_store.get_chunks_by_chapter_topic(chapter, topic)
            
            # Add retrieval metadata
            for chunk in chunks:
                chunk["retrieval_method"] = "direct"
                chunk["retrieval_score"] = 1.0
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Direct retrieval failed: {e}")
            return []
    
    def _retrieve_by_semantic_search(self, query_text: str, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Retrieve chunks using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query_text)
            
            # Search with filters
            filters = {"chapter": chapter, "topic": topic}
            results = self.vector_store.search(query_embedding, k=self.default_k * 2, filters=filters)
            
            # Add retrieval metadata
            for result in results:
                result["retrieval_method"] = "semantic"
                result["query_text"] = query_text
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return []
    
    def _retrieve_with_boundary_detection(self, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Retrieve chunks with dynamic boundary detection."""
        try:
            # Get all chunks for the chapter
            chapter_chunks = [
                chunk for chunk in self.vector_store.chunk_metadata
                if chunk.get("metadata", {}).get("chapter") == chapter
            ]
            
            if not chapter_chunks:
                return []
            
            # Find topic boundaries
            topic_boundaries = self._detect_topic_boundaries(chapter_chunks, topic)
            
            # Retrieve chunks within boundaries
            boundary_results = []
            for start_idx, end_idx in topic_boundaries:
                boundary_chunks = chapter_chunks[start_idx:end_idx + 1]
                
                for chunk in boundary_chunks:
                    chunk_copy = chunk.copy()
                    chunk_copy["retrieval_method"] = "boundary"
                    chunk_copy["boundary_start"] = start_idx
                    chunk_copy["boundary_end"] = end_idx
                    boundary_results.append(chunk_copy)
            
            return boundary_results
            
        except Exception as e:
            logger.warning(f"Boundary detection retrieval failed: {e}")
            return []
    
    def _detect_topic_boundaries(self, chunks: List[Dict[str, Any]], target_topic: str) -> List[Tuple[int, int]]:
        """Detect topic boundaries in chunks."""
        boundaries = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            
            # Check for topic patterns
            for pattern in self.topic_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    
                    # Check if this matches our target topic
                    if self._matches_topic(matched_text, target_topic):
                        # Find the end of this topic
                        end_idx = self._find_topic_end(chunks, i, target_topic)
                        boundaries.append((i, end_idx))
                        break
        
        return boundaries
    
    def _matches_topic(self, text: str, target_topic: str) -> bool:
        """Check if text matches the target topic."""
        # Normalize text
        text_lower = text.lower().strip()
        target_lower = target_topic.lower().strip()
        
        # Direct match
        if target_lower in text_lower:
            return True
        
        # Pattern match (e.g., "30.3" matches "30.3 Conditional Probability")
        if re.search(rf'\b{re.escape(target_lower)}\b', text_lower):
            return True
        
        return False
    
    def _find_topic_end(self, chunks: List[Dict[str, Any]], start_idx: int, current_topic: str) -> int:
        """Find the end of a topic section."""
        for i in range(start_idx + 1, len(chunks)):
            content = chunks[i].get("content", "")
            
            # Check for next topic
            for pattern in self.topic_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return i - 1
            
            # Check for end of chapter
            if "Chapter" in content and i > start_idx + 2:
                return i - 1
        
        return len(chunks) - 1
    
    def _combine_retrieval_results(self, direct_results: List[Dict[str, Any]], 
                                 semantic_results: List[Dict[str, Any]], 
                                 boundary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from different retrieval methods."""
        combined = {}
        
        # Add direct results
        for result in direct_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                combined[chunk_id] = result
        
        # Add semantic results (may override direct results with better scores)
        for result in semantic_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                if chunk_id not in combined or result.get("similarity_score", 0) > combined[chunk_id].get("similarity_score", 0):
                    combined[chunk_id] = result
        
        # Add boundary results
        for result in boundary_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in combined:
                combined[chunk_id] = result
        
        return list(combined.values())
    
    def _filter_and_rank_results(self, results: List[Dict[str, Any]], 
                                chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Filter and rank retrieval results."""
        filtered = []
        
        for result in results:
            # Apply content type weighting
            chunk_type = result.get("chunk_type", "general")
            weight = self.content_weights.get(chunk_type, 0.5)
            
            # Calculate final score
            base_score = result.get("similarity_score", 0.5)
            final_score = base_score * weight
            
            # Apply similarity threshold
            if final_score >= self.similarity_threshold:
                result["final_score"] = final_score
                result["content_weight"] = weight
                filtered.append(result)
        
        # Sort by final score
        filtered.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return filtered
    
    def _limit_context_length(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit the total context length."""
        limited_results = []
        total_length = 0
        
        for result in results:
            content_length = len(result.get("content", ""))
            
            if total_length + content_length <= self.max_context_length:
                limited_results.append(result)
                total_length += content_length
            else:
                # Try to truncate the content
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # Minimum useful length
                    truncated_result = result.copy()
                    truncated_result["content"] = result["content"][:remaining_length] + "..."
                    limited_results.append(truncated_result)
                break
        
        return limited_results
    
    def retrieve_with_context_window(self, query_text: str, chapter: int, topic: str,
                                   window_size: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with context window around relevant content.
        
        Args:
            query_text: Query text
            chapter: Chapter number
            topic: Topic identifier
            window_size: Number of chunks to include around each relevant chunk
            
        Returns:
            List of chunks with context window
        """
        try:
            # Get initial results
            initial_results = self.retrieve_for_question_extraction(chapter, topic, query_text)
            
            if not initial_results:
                return []
            
            # Get all chunks for the chapter
            all_chunks = self.vector_store.get_chunks_by_chapter_topic(chapter, topic)
            
            # Create context windows
            context_results = []
            used_indices = set()
            
            for result in initial_results:
                # Find the index of this chunk in all chunks
                chunk_index = None
                for i, chunk in enumerate(all_chunks):
                    if chunk.get("chunk_id") == result.get("chunk_id"):
                        chunk_index = i
                        break
                
                if chunk_index is not None:
                    # Add context window
                    start_idx = max(0, chunk_index - window_size)
                    end_idx = min(len(all_chunks) - 1, chunk_index + window_size)
                    
                    for i in range(start_idx, end_idx + 1):
                        if i not in used_indices:
                            context_chunk = all_chunks[i].copy()
                            context_chunk["context_window"] = True
                            context_chunk["center_chunk_id"] = result.get("chunk_id")
                            context_chunk["distance_from_center"] = abs(i - chunk_index)
                            context_results.append(context_chunk)
                            used_indices.add(i)
            
            # Sort by distance from center and relevance
            context_results.sort(key=lambda x: (
                x.get("distance_from_center", 0),
                -x.get("final_score", 0)
            ))
            
            return context_results
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to retrieve with context window: {str(e)}",
                component="retriever"
            )
    
    def retrieve_questions_only(self, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Retrieve only question-related chunks."""
        try:
            # Get all chunks for the chapter/topic
            all_chunks = self.vector_store.get_chunks_by_chapter_topic(chapter, topic)
            
            # Filter for question-related content
            question_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get("chunk_type", "")
                content = chunk.get("content", "").lower()
                
                # Check if it's question-related
                is_question = (
                    chunk_type in ["question", "exercise", "illustration", "example"] or
                    any(keyword in content for keyword in ["find", "calculate", "determine", "solve", "prove"]) or
                    re.search(r'\d+\.\s*[A-Z]', content)  # Numbered questions
                )
                
                if is_question:
                    chunk_copy = chunk.copy()
                    chunk_copy["question_relevance"] = True
                    question_chunks.append(chunk_copy)
            
            # Sort by relevance
            question_chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return question_chunks
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to retrieve questions: {str(e)}",
                component="retriever"
            )
    
    def get_retrieval_statistics(self, chapter: int, topic: str) -> Dict[str, Any]:
        """Get statistics about retrieval for a chapter/topic."""
        try:
            chunks = self.vector_store.get_chunks_by_chapter_topic(chapter, topic)
            
            if not chunks:
                return {"status": "no_chunks_found"}
            
            # Calculate statistics
            total_chunks = len(chunks)
            chunk_types = defaultdict(int)
            content_lengths = []
            
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "unknown")
                chunk_types[chunk_type] += 1
                content_lengths.append(len(chunk.get("content", "")))
            
            stats = {
                "total_chunks": total_chunks,
                "chunk_type_distribution": dict(chunk_types),
                "average_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
                "total_content_length": sum(content_lengths),
                "chapter": chapter,
                "topic": topic,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get retrieval statistics: {e}")
            return {"error": str(e)} 