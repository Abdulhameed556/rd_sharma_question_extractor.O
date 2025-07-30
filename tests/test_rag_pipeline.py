"""
Unit tests for RAG pipeline module.

This module tests embeddings generation, content chunking, vector storage,
and retrieval functionality.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.rag_pipeline.embeddings import EmbeddingGenerator
from src.rag_pipeline.chunker import ContentChunker
from src.rag_pipeline.vector_store import VectorStore
from src.rag_pipeline.retriever import RAGRetriever
from src.utils.exceptions import RAGPipelineError


class TestEmbeddingGenerator:
    """Test embedding generator functionality."""

    def test_initialization(self, test_config):
        """Test embedding generator initialization."""
        generator = EmbeddingGenerator(test_config)
        assert generator.config == test_config
        assert generator.model is None

    @patch('src.rag_pipeline.embeddings.SentenceTransformer')
    def test_initialize_model(self, mock_sentence_transformer, test_config):
        """Test model initialization."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(test_config)
        generator._initialize_model()
        
        assert generator.model == mock_model
        mock_sentence_transformer.assert_called_once()

    @patch('src.rag_pipeline.embeddings.SentenceTransformer')
    def test_generate_embedding(self, mock_sentence_transformer, test_config):
        """Test single embedding generation."""
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_sentence_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(test_config)
        generator._initialize_model()
        
        text = "Find P(A|B) where A and B are events."
        embedding = generator.generate_embedding(text)
        
        assert len(embedding) == 5
        assert isinstance(embedding, np.ndarray)
        mock_model.encode.assert_called_once()

    @patch('src.rag_pipeline.embeddings.SentenceTransformer')
    def test_generate_embeddings_batch(self, mock_sentence_transformer, test_config):
        """Test batch embedding generation."""
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ])
        mock_sentence_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(test_config)
        generator._initialize_model()
        
        texts = [
            "Find P(A|B) where A and B are events.",
            "Calculate the probability of drawing two red balls."
        ]
        
        embeddings = generator.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 5 for emb in embeddings)
        mock_model.encode.assert_called_once()

    def test_preprocess_text(self, test_config):
        """Test text preprocessing."""
        generator = EmbeddingGenerator(test_config)
        
        text = "  Find   P(A|B)   where   A   and   B   are   events.  "
        processed = generator._preprocess_text(text)
        
        assert processed == "Find P(A|B) where A and B are events."
        assert "  " not in processed  # No double spaces

    def test_normalize_math_expressions(self, test_config):
        """Test mathematical expression normalization."""
        generator = EmbeddingGenerator(test_config)
        
        text = "Find P(A|B) where P(A) = 1/2 and P(B) = 1/3"
        normalized = generator._normalize_math_expressions(text)
        
        # Should preserve mathematical expressions
        assert "P(A|B)" in normalized
        assert "1/2" in normalized
        assert "1/3" in normalized

    def test_calculate_similarity(self, test_config):
        """Test cosine similarity calculation."""
        generator = EmbeddingGenerator(test_config)
        
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        # Orthogonal vectors should have similarity 0
        similarity_12 = generator.calculate_similarity(vec1, vec2)
        assert abs(similarity_12) < 0.1
        
        # Identical vectors should have similarity 1
        similarity_13 = generator.calculate_similarity(vec1, vec3)
        assert abs(similarity_13 - 1.0) < 0.1

    def test_find_similar_chunks(self, test_config):
        """Test finding similar chunks."""
        generator = EmbeddingGenerator(test_config)
        
        # Mock embeddings
        query_embedding = np.array([1, 0, 0])
        chunk_embeddings = {
            'chunk1': np.array([0.9, 0.1, 0]),
            'chunk2': np.array([0.1, 0.9, 0]),
            'chunk3': np.array([0.8, 0.2, 0])
        }
        
        similar = generator.find_similar_chunks(query_embedding, chunk_embeddings, top_k=2)
        
        assert len(similar) == 2
        assert similar[0][0] == 'chunk1'  # Most similar
        assert similar[1][0] == 'chunk3'  # Second most similar

    def test_save_and_load_cache(self, test_config, temp_output_dir):
        """Test embedding cache functionality."""
        generator = EmbeddingGenerator(test_config)
        generator.cache_dir = temp_output_dir
        
        # Test data
        test_embeddings = {
            'text1': np.array([0.1, 0.2, 0.3]),
            'text2': np.array([0.4, 0.5, 0.6])
        }
        
        # Save cache
        generator.save_embeddings_cache(test_embeddings, 'test_cache')
        
        # Load cache
        loaded_embeddings = generator.load_embeddings_cache('test_cache')
        assert len(loaded_embeddings) == 2
        assert 'text1' in loaded_embeddings


class TestContentChunker:
    """Test content chunker functionality."""

    def test_initialization(self, test_config):
        """Test content chunker initialization."""
        chunker = ContentChunker(test_config)
        assert chunker.config == test_config

    def test_chunk_content(self, test_config):
        """Test content chunking."""
        chunker = ContentChunker(test_config)
        
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
        
        chunks = chunker.chunk_content(content)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)

    def test_preprocess_content(self, test_config):
        """Test content preprocessing."""
        chunker = ContentChunker(test_config)
        
        content = "  Find   P(A|B)   where   A   and   B   are   events.  "
        processed = chunker._preprocess_content(content)
        
        assert processed == "Find P(A|B) where A and B are events."
        assert "  " not in processed

    def test_clean_math_expressions(self, test_config):
        """Test mathematical expression cleaning."""
        chunker = ContentChunker(test_config)
        
        content = "Find P(A|B) where P(A) = 1/2 and P(B) = 1/3"
        cleaned = chunker._clean_math_expressions(content)
        
        # Should preserve mathematical expressions
        assert "P(A|B)" in cleaned
        assert "1/2" in cleaned
        assert "1/3" in cleaned

    def test_analyze_content_structure(self, test_config):
        """Test content structure analysis."""
        chunker = ContentChunker(test_config)
        
        content = """
        Chapter 30: Probability
        30.3 Conditional Probability
        Theory: Some theory here...
        Illustration 1: Example problem...
        Exercise 30.3
        1. Question one...
        2. Question two...
        """
        
        structure = chunker._analyze_content_structure(content)
        
        assert structure['paragraphs'] > 0
        assert structure['sentences'] > 0
        assert structure['math_density'] >= 0
        assert 'illustration' in structure['content_types']
        assert 'exercise' in structure['content_types']

    def test_chunk_by_structure(self, test_config):
        """Test structural chunking."""
        chunker = ContentChunker(test_config)
        
        content = """
        Chapter 30: Probability
        
        30.3 Conditional Probability
        
        Theory: Conditional probability is defined as...
        
        Illustration 1: A bag contains 4 red balls...
        
        Exercise 30.3
        
        1. A die is thrown twice...
        2. In a class of 60 students...
        """
        
        chunks = chunker._chunk_by_structure(content)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'content') for chunk in chunks)

    def test_chunk_by_size(self, test_config):
        """Test size-based chunking."""
        chunker = ContentChunker(test_config)
        
        content = "This is a long text that needs to be chunked into smaller pieces. " * 10
        
        chunks = chunker._chunk_by_size(content)
        
        assert len(chunks) > 0
        assert all(len(chunk.content) <= chunker.config.chunk_size for chunk in chunks)

    def test_create_chunk(self, test_config):
        """Test chunk creation."""
        chunker = ContentChunker(test_config)
        
        content = "Sample chunk content"
        metadata = {'page': 1, 'chapter': '30', 'topic': '30.3'}
        
        chunk = chunker._create_chunk(content, metadata)
        
        assert chunk.content == content
        assert chunk.metadata == metadata
        assert hasattr(chunk, 'id')

    def test_determine_chunk_type(self, test_config):
        """Test chunk type determination."""
        chunker = ContentChunker(test_config)
        
        # Test different content types
        assert chunker._determine_chunk_type("Illustration 1: ...") == "illustration"
        assert chunker._determine_chunk_type("Exercise 30.3") == "exercise"
        assert chunker._determine_chunk_type("Theory: ...") == "theory"
        assert chunker._determine_chunk_type("Regular text") == "content"

    def test_merge_small_chunks(self, test_config):
        """Test small chunk merging."""
        chunker = ContentChunker(test_config)
        
        chunks = [
            chunker._create_chunk("Small chunk 1", {}),
            chunker._create_chunk("Small chunk 2", {}),
            chunker._create_chunk("Large chunk with much more content that exceeds the minimum size threshold", {})
        ]
        
        merged = chunker.merge_small_chunks(chunks, min_size=20)
        
        assert len(merged) <= len(chunks)
        assert all(len(chunk.content) >= 20 or chunk.content_type == 'content' for chunk in merged)

    def test_split_large_chunks(self, test_config):
        """Test large chunk splitting."""
        chunker = ContentChunker(test_config)
        
        large_content = "This is a very large chunk of content. " * 20
        chunk = chunker._create_chunk(large_content, {})
        
        split_chunks = chunker.split_large_chunks([chunk], max_size=100)
        
        assert len(split_chunks) > 1
        assert all(len(chunk.content) <= 100 for chunk in split_chunks)


class TestVectorStore:
    """Test vector store functionality."""

    def test_initialization(self, test_config):
        """Test vector store initialization."""
        store = VectorStore(test_config)
        assert store.config == test_config
        assert store.index is None
        assert store.metadata == {}

    def test_initialize_index(self, test_config):
        """Test index initialization."""
        store = VectorStore(test_config)
        
        store.initialize_index(dimension=5, index_type='flat')
        
        assert store.index is not None
        assert store.dimension == 5

    def test_add_chunks(self, test_config):
        """Test adding chunks to vector store."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Sample content 1',
                'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'metadata': {'page': 1, 'chapter': '30'}
            },
            {
                'id': 'chunk2',
                'content': 'Sample content 2',
                'embedding': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'metadata': {'page': 2, 'chapter': '30'}
            }
        ]
        
        store.add_chunks(chunks)
        
        assert len(store.metadata) == 2
        assert 'chunk1' in store.metadata
        assert 'chunk2' in store.metadata

    def test_search(self, test_config):
        """Test vector search functionality."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunks
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Sample content 1',
                'embedding': np.array([1, 0, 0, 0, 0]),
                'metadata': {'page': 1, 'chapter': '30'}
            },
            {
                'id': 'chunk2',
                'content': 'Sample content 2',
                'embedding': np.array([0, 1, 0, 0, 0]),
                'metadata': {'page': 2, 'chapter': '30'}
            }
        ]
        store.add_chunks(chunks)
        
        # Search
        query_embedding = np.array([1, 0, 0, 0, 0])
        results = store.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert results[0]['id'] == 'chunk1'  # Most similar

    def test_search_by_content(self, test_config):
        """Test content-based search."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunks
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Probability question about balls',
                'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'metadata': {'page': 1, 'chapter': '30', 'content_type': 'question'}
            },
            {
                'id': 'chunk2',
                'content': 'Theory about conditional probability',
                'embedding': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'metadata': {'page': 2, 'chapter': '30', 'content_type': 'theory'}
            }
        ]
        store.add_chunks(chunks)
        
        # Search by content type
        results = store.search_by_content('question', filters={'content_type': 'question'})
        
        assert len(results) == 1
        assert results[0]['id'] == 'chunk1'

    def test_get_chunk_by_id(self, test_config):
        """Test retrieving chunk by ID."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunk
        chunk = {
            'id': 'chunk1',
            'content': 'Sample content',
            'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'metadata': {'page': 1, 'chapter': '30'}
        }
        store.add_chunks([chunk])
        
        # Retrieve by ID
        retrieved = store.get_chunk_by_id('chunk1')
        assert retrieved['content'] == 'Sample content'
        
        # Test non-existent ID
        retrieved = store.get_chunk_by_id('nonexistent')
        assert retrieved is None

    def test_get_chunks_by_type(self, test_config):
        """Test retrieving chunks by type."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunks
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Question 1',
                'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'metadata': {'content_type': 'question'}
            },
            {
                'id': 'chunk2',
                'content': 'Theory content',
                'embedding': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'metadata': {'content_type': 'theory'}
            }
        ]
        store.add_chunks(chunks)
        
        # Get questions
        questions = store.get_chunks_by_type('question')
        assert len(questions) == 1
        assert questions[0]['id'] == 'chunk1'

    def test_get_chunks_by_chapter_topic(self, test_config):
        """Test retrieving chunks by chapter and topic."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunks
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Chapter 30 content',
                'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'metadata': {'chapter': '30', 'topic': '30.3'}
            },
            {
                'id': 'chunk2',
                'content': 'Chapter 31 content',
                'embedding': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'metadata': {'chapter': '31', 'topic': '31.1'}
            }
        ]
        store.add_chunks(chunks)
        
        # Get chapter 30 chunks
        chapter_30_chunks = store.get_chunks_by_chapter_topic('30', '30.3')
        assert len(chapter_30_chunks) == 1
        assert chapter_30_chunks[0]['id'] == 'chunk1'

    def test_get_statistics(self, test_config):
        """Test getting vector store statistics."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunks
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Content 1',
                'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'metadata': {'content_type': 'question'}
            },
            {
                'id': 'chunk2',
                'content': 'Content 2',
                'embedding': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                'metadata': {'content_type': 'theory'}
            }
        ]
        store.add_chunks(chunks)
        
        stats = store.get_statistics()
        
        assert stats['total_chunks'] == 2
        assert stats['dimension'] == 5
        assert 'question' in stats['content_types']
        assert 'theory' in stats['content_types']

    def test_save_and_load_index(self, test_config, temp_output_dir):
        """Test index saving and loading."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunk
        chunk = {
            'id': 'chunk1',
            'content': 'Sample content',
            'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'metadata': {'page': 1, 'chapter': '30'}
        }
        store.add_chunks([chunk])
        
        # Save index
        index_path = Path(temp_output_dir) / 'vector_store'
        store.save_index(str(index_path))
        
        # Create new store and load index
        new_store = VectorStore(test_config)
        new_store.load_index(str(index_path))
        
        assert new_store.index is not None
        assert len(new_store.metadata) == 1
        assert 'chunk1' in new_store.metadata

    def test_clear(self, test_config):
        """Test clearing vector store."""
        store = VectorStore(test_config)
        store.initialize_index(dimension=5, index_type='flat')
        
        # Add test chunk
        chunk = {
            'id': 'chunk1',
            'content': 'Sample content',
            'embedding': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'metadata': {'page': 1, 'chapter': '30'}
        }
        store.add_chunks([chunk])
        
        # Clear store
        store.clear()
        
        assert len(store.metadata) == 0
        assert store.index is None


class TestRAGRetriever:
    """Test RAG retriever functionality."""

    def test_initialization(self, test_config):
        """Test RAG retriever initialization."""
        retriever = RAGRetriever(test_config)
        assert retriever.config == test_config
        assert retriever.vector_store is None

    def test_retrieve_for_question_extraction(self, test_config):
        """Test retrieval for question extraction."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store
        mock_store = Mock()
        mock_store.search.return_value = [
            {'id': 'chunk1', 'content': 'Question 1', 'score': 0.9},
            {'id': 'chunk2', 'content': 'Question 2', 'score': 0.8}
        ]
        retriever.vector_store = mock_store
        
        # Mock embedding generator
        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        retriever.embedding_generator = mock_embedding_gen
        
        query = "Find probability questions"
        results = retriever.retrieve_for_question_extraction(query, chapter='30', topic='30.3')
        
        assert len(results) == 2
        assert results[0]['score'] > results[1]['score']

    def test_retrieve_by_chapter_topic(self, test_config):
        """Test retrieval by chapter and topic."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store
        mock_store = Mock()
        mock_store.get_chunks_by_chapter_topic.return_value = [
            {'id': 'chunk1', 'content': 'Chapter 30 content', 'metadata': {'chapter': '30', 'topic': '30.3'}}
        ]
        retriever.vector_store = mock_store
        
        results = retriever._retrieve_by_chapter_topic('30', '30.3')
        
        assert len(results) == 1
        assert results[0]['id'] == 'chunk1'

    def test_retrieve_by_semantic_search(self, test_config):
        """Test semantic search retrieval."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store
        mock_store = Mock()
        mock_store.search.return_value = [
            {'id': 'chunk1', 'content': 'Relevant content', 'score': 0.9}
        ]
        retriever.vector_store = mock_store
        
        # Mock embedding generator
        mock_embedding_gen = Mock()
        mock_embedding_gen.generate_embedding.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        retriever.embedding_generator = mock_embedding_gen
        
        query = "probability questions"
        results = retriever._retrieve_by_semantic_search(query)
        
        assert len(results) == 1
        assert results[0]['score'] > 0.8

    def test_retrieve_with_boundary_detection(self, test_config):
        """Test retrieval with boundary detection."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store
        mock_store = Mock()
        mock_store.get_chunks_by_chapter_topic.return_value = [
            {'id': 'chunk1', 'content': 'Start of topic', 'metadata': {'page': 1}},
            {'id': 'chunk2', 'content': 'Middle content', 'metadata': {'page': 2}},
            {'id': 'chunk3', 'content': 'End of topic', 'metadata': {'page': 3}}
        ]
        retriever.vector_store = mock_store
        
        results = retriever._retrieve_with_boundary_detection('30', '30.3')
        
        assert len(results) == 3
        assert results[0]['metadata']['page'] == 1
        assert results[2]['metadata']['page'] == 3

    def test_detect_topic_boundaries(self, test_config):
        """Test topic boundary detection."""
        retriever = RAGRetriever(test_config)
        
        chunks = [
            {'content': 'Chapter 30: Probability', 'metadata': {'page': 1}},
            {'content': '30.3 Conditional Probability', 'metadata': {'page': 2}},
            {'content': 'Theory content', 'metadata': {'page': 3}},
            {'content': 'Exercise 30.3', 'metadata': {'page': 4}},
            {'content': 'Chapter 31: Statistics', 'metadata': {'page': 5}}
        ]
        
        boundaries = retriever._detect_topic_boundaries(chunks, '30', '30.3')
        
        assert boundaries['start_page'] == 2
        assert boundaries['end_page'] == 4

    def test_matches_topic(self, test_config):
        """Test topic matching."""
        retriever = RAGRetriever(test_config)
        
        # Test matching content
        assert retriever._matches_topic("30.3 Conditional Probability", '30', '30.3')
        assert retriever._matches_topic("Exercise 30.3", '30', '30.3')
        
        # Test non-matching content
        assert not retriever._matches_topic("Chapter 31: Statistics", '30', '30.3')
        assert not retriever._matches_topic("30.4 Independent Events", '30', '30.3')

    def test_find_topic_end(self, test_config):
        """Test finding topic end."""
        retriever = RAGRetriever(test_config)
        
        chunks = [
            {'content': '30.3 Conditional Probability', 'metadata': {'page': 1}},
            {'content': 'Theory content', 'metadata': {'page': 2}},
            {'content': 'Exercise 30.3', 'metadata': {'page': 3}},
            {'content': '30.4 Independent Events', 'metadata': {'page': 4}}
        ]
        
        end_page = retriever._find_topic_end(chunks, 1, '30', '30.3')
        assert end_page == 3

    def test_combine_retrieval_results(self, test_config):
        """Test combining retrieval results."""
        retriever = RAGRetriever(test_config)
        
        results1 = [{'id': 'chunk1', 'score': 0.9}]
        results2 = [{'id': 'chunk2', 'score': 0.8}]
        
        combined = retriever._combine_retrieval_results([results1, results2])
        
        assert len(combined) == 2
        assert combined[0]['score'] > combined[1]['score']

    def test_filter_and_rank_results(self, test_config):
        """Test filtering and ranking results."""
        retriever = RAGRetriever(test_config)
        
        results = [
            {'id': 'chunk1', 'score': 0.9, 'metadata': {'content_type': 'question'}},
            {'id': 'chunk2', 'score': 0.8, 'metadata': {'content_type': 'theory'}},
            {'id': 'chunk3', 'score': 0.7, 'metadata': {'content_type': 'question'}}
        ]
        
        filtered = retriever._filter_and_rank_results(results, filters={'content_type': 'question'})
        
        assert len(filtered) == 2
        assert all(r['metadata']['content_type'] == 'question' for r in filtered)
        assert filtered[0]['score'] > filtered[1]['score']

    def test_limit_context_length(self, test_config):
        """Test context length limiting."""
        retriever = RAGRetriever(test_config)
        
        results = [
            {'id': 'chunk1', 'content': 'Short content', 'score': 0.9},
            {'id': 'chunk2', 'content': 'Long content ' * 100, 'score': 0.8},
            {'id': 'chunk3', 'content': 'Medium content ' * 50, 'score': 0.7}
        ]
        
        limited = retriever._limit_context_length(results, max_tokens=1000)
        
        assert len(limited) <= len(results)
        total_length = sum(len(r['content']) for r in limited)
        assert total_length <= 1000

    def test_get_retrieval_statistics(self, test_config):
        """Test retrieval statistics."""
        retriever = RAGRetriever(test_config)
        
        # Mock vector store
        mock_store = Mock()
        mock_store.get_statistics.return_value = {
            'total_chunks': 100,
            'dimension': 5,
            'content_types': {'question': 50, 'theory': 30, 'illustration': 20}
        }
        retriever.vector_store = mock_store
        
        stats = retriever.get_retrieval_statistics()
        
        assert stats['total_chunks'] == 100
        assert stats['dimension'] == 5
        assert 'question' in stats['content_types'] 