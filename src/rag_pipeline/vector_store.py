"""
Vector store for RD Sharma Question Extractor.

This module provides FAISS-based vector storage and retrieval capabilities
for the RAG pipeline.
"""

import faiss
import numpy as np
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib

from ..utils.logger import get_logger
from ..utils.exceptions import RAGPipelineError
from ..config import config

logger = get_logger(__name__)


class VectorStore:
    """FAISS-based vector store for mathematical content."""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to save/load FAISS index
        """
        self.index_path = Path(index_path or config.vector_db_path)
        self.index = None
        self.chunk_metadata = []
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.is_initialized = False
        
        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Vector store initialized with index path: {self.index_path}")
    
    def initialize_index(self, embedding_dim: int = 384, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        try:
            self.embedding_dim = embedding_dim
            
            if index_type == "flat":
                # Simple flat index (good for small datasets)
                self.index = faiss.IndexFlatIP(embedding_dim)
            elif index_type == "ivf":
                # Inverted file index (good for larger datasets)
                nlist = min(100, max(1, embedding_dim // 4))  # Number of clusters
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            elif index_type == "hnsw":
                # Hierarchical navigable small world (good for approximate search)
                self.index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbors
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.is_initialized = True
            logger.info(f"FAISS index initialized: {index_type} with dim={embedding_dim}")
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to initialize FAISS index: {str(e)}",
                component="vector_store"
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunks with embeddings
        """
        try:
            if not chunks:
                return
            
            # Initialize index if not already done
            if not self.is_initialized:
                embedding_dim = len(chunks[0].get("embedding", []))
                if embedding_dim == 0:
                    raise RAGPipelineError("No embeddings found in chunks", component="vector_store")
                self.initialize_index(embedding_dim)
            
            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if embedding is not None:
                    embeddings.append(embedding)
                    metadata.append(chunk)
            
            if not embeddings:
                logger.warning("No valid embeddings found in chunks")
                return
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            if self.index.ntotal == 0:
                # First batch - add directly
                self.index.add(embeddings_array)
            else:
                # Subsequent batches - check if index needs training
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    # Train the index
                    self.index.train(embeddings_array)
                
                self.index.add(embeddings_array)
            
            # Store metadata
            self.chunk_metadata.extend(metadata)
            
            logger.info(f"Added {len(embeddings)} chunks to vector store. Total: {self.index.ntotal}")
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to add chunks to vector store: {str(e)}",
                component="vector_store",
                context={"chunk_count": len(chunks)}
            )
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar chunks with scores
        """
        try:
            if not self.is_initialized or self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Get results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunk_metadata):
                    chunk = self.chunk_metadata[idx].copy()
                    chunk["similarity_score"] = float(score)
                    chunk["rank"] = i + 1
                    
                    # Apply filters if provided
                    if filters and not self._matches_filters(chunk, filters):
                        continue
                    
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to search vector store: {str(e)}",
                component="vector_store"
            )
    
    def _matches_filters(self, chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if chunk matches the given filters."""
        for key, value in filters.items():
            if key in chunk.get("metadata", {}):
                chunk_value = chunk["metadata"][key]
                if isinstance(value, (list, tuple)):
                    if chunk_value not in value:
                        return False
                else:
                    if chunk_value != value:
                        return False
            else:
                return False
        return True
    
    def search_by_content(self, query_text: str, embedding_generator, k: int = 5,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search by text content (generates embedding automatically).
        
        Args:
            query_text: Query text
            embedding_generator: Embedding generator instance
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Generate embedding for query
            query_embedding = embedding_generator.generate_embedding(query_text)
            
            # Search vector store
            return self.search(query_embedding, k, filters)
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to search by content: {str(e)}",
                component="vector_store"
            )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by its ID."""
        for chunk in self.chunk_metadata:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_type(self, chunk_type: str) -> List[Dict[str, Any]]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunk_metadata if chunk.get("chunk_type") == chunk_type]
    
    def get_chunks_by_chapter_topic(self, chapter: int, topic: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific chapter and topic."""
        return [
            chunk for chunk in self.chunk_metadata
            if (chunk.get("metadata", {}).get("chapter") == chapter and
                chunk.get("metadata", {}).get("topic") == topic)
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        stats = {
            "total_chunks": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, 'is_trained', True),
        }
        
        # Chunk type distribution
        chunk_types = {}
        for chunk in self.chunk_metadata:
            chunk_type = chunk.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        stats["chunk_type_distribution"] = chunk_types
        
        # Chapter distribution
        chapters = {}
        for chunk in self.chunk_metadata:
            chapter = chunk.get("metadata", {}).get("chapter", "unknown")
            chapters[chapter] = chapters.get(chapter, 0) + 1
        
        stats["chapter_distribution"] = chapters
        
        return stats
    
    def save_index(self, path: Optional[str] = None):
        """Save FAISS index and metadata."""
        try:
            save_path = Path(path or self.index_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = save_path.with_suffix('.faiss')
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2, default=str)
            
            logger.info(f"Vector store saved to {save_path}")
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to save vector store: {str(e)}",
                component="vector_store"
            )
    
    def load_index(self, path: Optional[str] = None):
        """Load FAISS index and metadata."""
        try:
            load_path = Path(path or self.index_path)
            
            # Load FAISS index
            index_path = load_path.with_suffix('.faiss')
            if not index_path.exists():
                logger.warning(f"FAISS index not found: {index_path}")
                return False
            
            self.index = faiss.read_index(str(index_path))
            self.embedding_dim = self.index.d
            self.is_initialized = True
            
            # Load metadata
            metadata_path = load_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.chunk_metadata = json.load(f)
            else:
                self.chunk_metadata = []
            
            logger.info(f"Vector store loaded from {load_path}")
            return True
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to load vector store: {str(e)}",
                component="vector_store"
            )
    
    def clear(self):
        """Clear all data from the vector store."""
        if self.is_initialized:
            self.index.reset()
        self.chunk_metadata = []
        logger.info("Vector store cleared")
    
    def delete_chunks(self, chunk_ids: List[str]):
        """Delete specific chunks from the vector store."""
        try:
            # Find indices to remove
            indices_to_remove = []
            for i, chunk in enumerate(self.chunk_metadata):
                if chunk.get("chunk_id") in chunk_ids:
                    indices_to_remove.append(i)
            
            # Remove from metadata
            for i in reversed(indices_to_remove):
                del self.chunk_metadata[i]
            
            # Rebuild index
            if self.chunk_metadata:
                embeddings = [chunk.get("embedding") for chunk in self.chunk_metadata]
                embeddings_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                
                # Create new index
                self.initialize_index(self.embedding_dim)
                self.index.add(embeddings_array)
            else:
                self.clear()
            
            logger.info(f"Deleted {len(indices_to_remove)} chunks from vector store")
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to delete chunks: {str(e)}",
                component="vector_store"
            )
    
    def optimize_index(self):
        """Optimize the index for better performance."""
        try:
            if not self.is_initialized:
                return
            
            # For IVF index, optimize the number of clusters
            if hasattr(self.index, 'nlist'):
                current_nlist = self.index.nlist
                optimal_nlist = min(100, max(1, self.index.ntotal // 100))
                
                if current_nlist != optimal_nlist:
                    logger.info(f"Optimizing index: nlist {current_nlist} -> {optimal_nlist}")
                    # This would require rebuilding the index with optimal parameters
                    # For now, just log the optimization opportunity
            
            logger.info("Index optimization completed")
            
        except Exception as e:
            logger.warning(f"Index optimization failed: {e}")
    
    def get_similar_chunks_batch(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Batch of query embeddings
            k: Number of results per query
            
        Returns:
            List of results for each query
        """
        try:
            if not self.is_initialized or self.index.ntotal == 0:
                return [[] for _ in range(len(query_embeddings))]
            
            # Normalize query embeddings
            query_embeddings = query_embeddings.astype(np.float32)
            faiss.normalize_L2(query_embeddings)
            
            # Perform batch search
            scores, indices = self.index.search(query_embeddings, min(k, self.index.ntotal))
            
            # Process results
            all_results = []
            for query_scores, query_indices in zip(scores, indices):
                query_results = []
                for i, (score, idx) in enumerate(zip(query_scores, query_indices)):
                    if idx < len(self.chunk_metadata):
                        chunk = self.chunk_metadata[idx].copy()
                        chunk["similarity_score"] = float(score)
                        chunk["rank"] = i + 1
                        query_results.append(chunk)
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            raise RAGPipelineError(
                f"Failed to perform batch search: {str(e)}",
                component="vector_store"
            ) 