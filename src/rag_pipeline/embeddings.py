"""
Vector embeddings for RD Sharma Question Extractor.

This module provides embedding generation capabilities optimized for
mathematical content and textbook structure.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import hashlib
import json
from pathlib import Path
import torch

from ..utils.logger import get_logger
from ..utils.exceptions import RAGPipelineError
from ..config import config

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for mathematical content."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name or "all-MiniLM-L6-v2"  # Good balance of speed and quality
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Mathematical content specific models (fallback options)
        self.math_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "multi-qa-MiniLM-L6-cos-v1"
        ]
        
        self._initialize_model()
        logger.info(f"Embedding generator initialized with model: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Try to load the specified model
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}")
            
            # Try fallback models
            for fallback_model in self.math_models:
                if fallback_model != self.model_name:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        self.model = SentenceTransformer(fallback_model)
                        self.model_name = fallback_model
                        self.embedding_dim = self.model.get_sentence_embedding_dimension()
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to('cuda')
                        
                        logger.info(f"Successfully loaded fallback model: {fallback_model}")
                        break
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
                        continue
            
            if self.model is None:
                raise RAGPipelineError(
                    f"Failed to initialize any embedding model: {str(e)}",
                    component="embeddings"
                )
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Vector embedding as numpy array
        """
        try:
            if not text.strip():
                # Return zero vector for empty text
                return np.zeros(self.embedding_dim)
            
            # Preprocess text for mathematical content
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            
            return embedding
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to generate embedding: {str(e)}",
                component="embeddings",
                context={"text_length": len(text)}
            )
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Matrix of embeddings (n_texts x embedding_dim)
        """
        try:
            if not texts:
                return np.array([])
            
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Filter out empty texts
            valid_texts = [text for text in processed_texts if text.strip()]
            valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
            
            if not valid_texts:
                # Return zero vectors for all texts
                return np.zeros((len(texts), self.embedding_dim))
            
            # Generate embeddings for valid texts
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            
            # Create full embedding matrix
            full_embeddings = np.zeros((len(texts), self.embedding_dim))
            for i, idx in enumerate(valid_indices):
                full_embeddings[idx] = embeddings[i]
            
            return full_embeddings
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to generate batch embeddings: {str(e)}",
                component="embeddings",
                context={"batch_size": len(texts)}
            )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Normalize mathematical expressions
        text = self._normalize_math_expressions(text)
        
        # Truncate if too long (model context limit)
        max_length = 512  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def _normalize_math_expressions(self, text: str) -> str:
        """
        Normalize mathematical expressions for better embedding.
        
        Args:
            text: Text with mathematical expressions
            
        Returns:
            Normalized text
        """
        import re
        
        # Normalize common mathematical symbols
        replacements = {
            r'\b(\d+)\s*[xX]\s*(\d+)\b': r'\1 × \2',  # Multiplication
            r'\b(\d+)\s*/\s*(\d+)\b': r'\1 ÷ \2',     # Division
            r'\b(\d+)\s*=\s*(\d+)\b': r'\1 = \2',     # Equality
            r'\b(\d+)\s*\+\s*(\d+)\b': r'\1 + \2',    # Addition
            r'\b(\d+)\s*-\s*(\d+)\b': r'\1 - \2',     # Subtraction
        }
        
        normalized = text
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for content chunks with metadata.
        
        Args:
            chunks: List of content chunks with metadata
            
        Returns:
            Chunks with embeddings added
        """
        try:
            # Extract text content from chunks
            texts = [chunk.get("content", "") for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
                chunk["embedding_dim"] = self.embedding_dim
                chunk["embedding_model"] = self.model_name
                
                # Generate content hash for caching
                content_hash = hashlib.md5(
                    chunk.get("content", "").encode()
                ).hexdigest()
                chunk["content_hash"] = content_hash
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to generate chunk embeddings: {str(e)}",
                component="embeddings",
                context={"chunk_count": len(chunks)}
            )
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_similar_chunks(
        self, 
        query_embedding: np.ndarray, 
        chunk_embeddings: List[Dict[str, Any]], 
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding
            chunk_embeddings: List of chunk embeddings with metadata
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            similarities = []
            
            for chunk in chunk_embeddings:
                chunk_embedding = np.array(chunk.get("embedding", []))
                
                if len(chunk_embedding) == 0:
                    continue
                
                similarity = self.calculate_similarity(query_embedding, chunk_embedding)
                
                if similarity >= similarity_threshold:
                    similarities.append({
                        "chunk": chunk,
                        "similarity": similarity
                    })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return top_k results
            return similarities[:top_k]
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to find similar chunks: {str(e)}",
                component="embeddings"
            )
    
    def save_embeddings_cache(self, cache_key: str, embeddings: List[Dict[str, Any]]):
        """Save embeddings to cache."""
        cache_file = Path(config.embeddings_cache_dir) / f"{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = []
        for embedding_data in embeddings:
            serializable = embedding_data.copy()
            if "embedding" in serializable:
                serializable["embedding"] = serializable["embedding"].tolist() if isinstance(serializable["embedding"], np.ndarray) else serializable["embedding"]
            serializable_embeddings.append(serializable)
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_embeddings, f, indent=2)
        
        logger.debug(f"Saved embeddings cache: {cache_file}")
    
    def load_embeddings_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load embeddings from cache."""
        cache_file = Path(config.embeddings_cache_dir) / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                embeddings = json.load(f)
            
            # Convert lists back to numpy arrays
            for embedding_data in embeddings:
                if "embedding" in embedding_data:
                    embedding_data["embedding"] = np.array(embedding_data["embedding"])
            
            logger.debug(f"Loaded embeddings cache: {cache_file}")
            return embeddings
        
        return None
    
    def generate_cache_key(self, content_hash: str) -> str:
        """Generate cache key for content."""
        return f"embeddings_{content_hash}_{self.model_name.replace('/', '_')}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_sequence_length": 512
        } 