"""
Unit tests for document processor module.

This module tests PDF handling, OCR processing, document indexing,
and content parsing functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from src.document_processor.pdf_handler import PDFHandler
from src.document_processor.ocr_processor import OCRProcessor
from src.document_processor.document_indexer import DocumentIndexer
from src.document_processor.content_parser import ContentParser
from src.utils.exceptions import DocumentProcessingError


class TestPDFHandler:
    """Test PDF handler functionality."""

    def test_initialization(self, test_config):
        """Test PDF handler initialization."""
        handler = PDFHandler(test_config)
        assert handler.config == test_config
        assert handler.document is None

    @patch('src.document_processor.pdf_handler.fitz')
    def test_load_document_success(self, mock_fitz, test_config):
        """Test successful PDF document loading."""
        # Mock document
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_fitz.open.return_value = mock_doc
        
        handler = PDFHandler(test_config)
        result = handler.load_document("test.pdf")
        
        assert result is True
        assert handler.document == mock_doc
        assert handler.page_count == 5

    @patch('src.document_processor.pdf_handler.fitz')
    def test_load_document_failure(self, mock_fitz, test_config):
        """Test PDF document loading failure."""
        mock_fitz.open.side_effect = Exception("File not found")
        
        handler = PDFHandler(test_config)
        
        with pytest.raises(DocumentProcessingError):
            handler.load_document("nonexistent.pdf")

    @patch('src.document_processor.pdf_handler.fitz')
    def test_extract_page_text(self, mock_fitz, test_config):
        """Test page text extraction."""
        # Mock page
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample page text"
        
        # Mock document
        mock_doc = Mock()
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        handler = PDFHandler(test_config)
        handler.load_document("test.pdf")
        
        text = handler.extract_page_text(0)
        assert text == "Sample page text"

    @patch('src.document_processor.pdf_handler.fitz')
    def test_extract_page_image(self, mock_fitz, test_config):
        """Test page image extraction."""
        # Mock page
        mock_page = Mock()
        mock_page.get_pixmap.return_value = Mock()
        mock_page.get_pixmap.return_value.tobytes.return_value = b"fake_image_data"
        
        # Mock document
        mock_doc = Mock()
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        handler = PDFHandler(test_config)
        handler.load_document("test.pdf")
        
        image = handler.extract_page_image(0)
        assert image is not None

    def test_detect_text_quality(self, test_config):
        """Test text quality detection."""
        handler = PDFHandler(test_config)
        
        # Test high quality text
        high_quality = "This is clear, readable text with proper formatting."
        quality_score = handler.detect_text_quality(high_quality)
        assert quality_score > 0.7
        
        # Test low quality text
        low_quality = "Th1s 1s garb13d t3xt w1th numb3rs"
        quality_score = handler.detect_text_quality(low_quality)
        assert quality_score < 0.5

    def test_extract_page_range(self, test_config):
        """Test page range extraction."""
        with patch.object(PDFHandler, 'extract_page_text') as mock_extract:
            mock_extract.return_value = "Page content"
            
            handler = PDFHandler(test_config)
            pages = handler.extract_page_range(0, 2)
            
            assert len(pages) == 3
            assert all(page['content'] == "Page content" for page in pages)

    def test_close_document(self, test_config):
        """Test document closing."""
        handler = PDFHandler(test_config)
        handler.document = Mock()
        
        handler.close()
        assert handler.document is None


class TestOCRProcessor:
    """Test OCR processor functionality."""

    def test_initialization(self, test_config):
        """Test OCR processor initialization."""
        processor = OCRProcessor(test_config)
        assert processor.config == test_config
        assert processor.reader is None

    @patch('src.document_processor.ocr_processor.easyocr')
    def test_initialize_reader(self, mock_easyocr, test_config):
        """Test OCR reader initialization."""
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        
        processor = OCRProcessor(test_config)
        processor._initialize_reader()
        
        assert processor.reader == mock_reader
        mock_easyocr.Reader.assert_called_once()

    @patch('src.document_processor.ocr_processor.easyocr')
    def test_process_image(self, mock_easyocr, test_config):
        """Test image processing with OCR."""
        # Mock OCR reader
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Sample text", 0.9)
        ]
        mock_easyocr.Reader.return_value = mock_reader
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        processor = OCRProcessor(test_config)
        result = processor.process_image(test_image)
        
        assert result['text'] == "Sample text"
        assert result['confidence'] > 0.8

    def test_preprocess_image(self, test_config):
        """Test image preprocessing."""
        processor = OCRProcessor(test_config)
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        processed = processor._preprocess_image(test_image)
        assert processed is not None
        assert processed.shape == (100, 100)  # Grayscale

    def test_post_process_results(self, test_config):
        """Test OCR results post-processing."""
        processor = OCRProcessor(test_config)
        
        raw_results = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "P(A|B)", 0.9),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "4 red balls", 0.8)
        ]
        
        processed = processor._post_process_results(raw_results)
        
        assert len(processed) == 2
        assert processed[0]['text'] == "P(A|B)"
        assert processed[0]['confidence'] == 0.9

    def test_correct_mathematical_symbols(self, test_config):
        """Test mathematical symbol correction."""
        processor = OCRProcessor(test_config)
        
        # Test common OCR errors
        corrected = processor._correct_mathematical_symbols("P(AIB)")  # I instead of |
        assert "|" in corrected
        
        corrected = processor._correct_mathematical_symbols("4 red bal1s")  # 1 instead of l
        assert "balls" in corrected

    def test_is_mathematical_expression(self, test_config):
        """Test mathematical expression detection."""
        processor = OCRProcessor(test_config)
        
        # Test mathematical expressions
        assert processor._is_mathematical_expression("P(A|B)")
        assert processor._is_mathematical_expression("4 + 5 = 9")
        assert processor._is_mathematical_expression("∫ f(x) dx")
        
        # Test non-mathematical text
        assert not processor._is_mathematical_expression("This is regular text")

    def test_extract_mathematical_content(self, test_config):
        """Test mathematical content extraction."""
        processor = OCRProcessor(test_config)
        
        text = "Find P(A|B) where A and B are events. Also calculate 4 + 5."
        math_content = processor._extract_mathematical_content(text)
        
        assert "P(A|B)" in math_content
        assert "4 + 5" in math_content

    def test_calculate_confidence(self, test_config):
        """Test confidence calculation."""
        processor = OCRProcessor(test_config)
        
        results = [
            {'confidence': 0.9},
            {'confidence': 0.8},
            {'confidence': 0.7}
        ]
        
        avg_confidence = processor._calculate_confidence(results)
        assert 0.7 < avg_confidence < 0.9

    def test_save_and_load_cache(self, test_config, temp_output_dir):
        """Test OCR cache functionality."""
        processor = OCRProcessor(test_config)
        processor.cache_dir = temp_output_dir
        
        # Test data
        test_data = {
            'image_hash': 'abc123',
            'results': [{'text': 'test', 'confidence': 0.9}]
        }
        
        # Save cache
        processor.save_cache(test_data)
        
        # Load cache
        loaded_data = processor.load_cache('abc123')
        assert loaded_data['results'][0]['text'] == 'test'


class TestDocumentIndexer:
    """Test document indexer functionality."""

    def test_initialization(self, test_config):
        """Test document indexer initialization."""
        indexer = DocumentIndexer(test_config)
        assert indexer.config == test_config
        assert indexer.document_index == {}

    def test_extract_table_of_contents(self, test_config):
        """Test table of contents extraction."""
        indexer = DocumentIndexer(test_config)
        
        # Mock PDF content
        content = """
        Chapter 30: Probability
        30.1 Introduction
        30.2 Recapitulation
        30.3 Conditional Probability
        30.4 Independent Events
        """
        
        toc = indexer.extract_table_of_contents(content)
        assert len(toc) > 0
        assert any('30.3' in item for item in toc)

    def test_detect_chapter_boundaries(self, test_config):
        """Test chapter boundary detection."""
        indexer = DocumentIndexer(test_config)
        
        # Mock page content
        pages = [
            {'page': 0, 'content': 'Chapter 30: Probability'},
            {'page': 1, 'content': '30.1 Introduction'},
            {'page': 2, 'content': '30.2 Recapitulation'},
            {'page': 3, 'content': '30.3 Conditional Probability'},
            {'page': 4, 'content': 'Chapter 31: Statistics'}
        ]
        
        boundaries = indexer.detect_chapter_boundaries(pages)
        assert len(boundaries) > 0
        assert any(b['chapter'] == '30' for b in boundaries)

    def test_detect_topic_boundaries(self, test_config):
        """Test topic boundary detection."""
        indexer = DocumentIndexer(test_config)
        
        # Mock chapter content
        chapter_content = """
        30.3 Conditional Probability
        
        Theory: Conditional probability is defined as...
        
        Illustration 1: A bag contains 4 red balls...
        
        Exercise 30.3
        
        1. A die is thrown twice...
        2. In a class of 60 students...
        """
        
        topics = indexer.detect_topic_boundaries(chapter_content, '30')
        assert len(topics) > 0
        assert any(t['topic'] == '30.3' for t in topics)

    def test_create_page_mapping(self, test_config):
        """Test page to content mapping creation."""
        indexer = DocumentIndexer(test_config)
        
        # Mock data
        chapters = [{'chapter': '30', 'start_page': 0, 'end_page': 10}]
        topics = [{'topic': '30.3', 'start_page': 5, 'end_page': 8}]
        
        mapping = indexer.create_page_mapping(chapters, topics)
        assert len(mapping) > 0
        assert mapping[5]['chapter'] == '30'
        assert mapping[5]['topic'] == '30.3'

    def test_save_and_load_index(self, test_config, temp_output_dir):
        """Test index saving and loading."""
        indexer = DocumentIndexer(test_config)
        indexer.index_path = Path(temp_output_dir) / 'index.json'
        
        # Test data
        test_index = {
            'chapters': [{'chapter': '30', 'start_page': 0, 'end_page': 10}],
            'topics': [{'topic': '30.3', 'start_page': 5, 'end_page': 8}]
        }
        
        # Save index
        indexer.save_index(test_index)
        
        # Load index
        loaded_index = indexer.load_index()
        assert loaded_index['chapters'][0]['chapter'] == '30'

    def test_get_chapter_pages(self, test_config):
        """Test chapter page retrieval."""
        indexer = DocumentIndexer(test_config)
        indexer.document_index = {
            'chapters': [{'chapter': '30', 'start_page': 0, 'end_page': 10}]
        }
        
        pages = indexer.get_chapter_pages('30')
        assert pages['start_page'] == 0
        assert pages['end_page'] == 10

    def test_get_topic_pages(self, test_config):
        """Test topic page retrieval."""
        indexer = DocumentIndexer(test_config)
        indexer.document_index = {
            'topics': [{'topic': '30.3', 'start_page': 5, 'end_page': 8}]
        }
        
        pages = indexer.get_topic_pages('30.3')
        assert pages['start_page'] == 5
        assert pages['end_page'] == 8


class TestContentParser:
    """Test content parser functionality."""

    def test_initialization(self, test_config):
        """Test content parser initialization."""
        parser = ContentParser(test_config)
        assert parser.config == test_config

    def test_classify_content_type(self, test_config):
        """Test content type classification."""
        parser = ContentParser(test_config)
        
        # Test different content types
        assert parser.classify_content_type("Illustration 1: A bag contains...") == "illustration"
        assert parser.classify_content_type("Exercise 30.3") == "exercise"
        assert parser.classify_content_type("Theory: Conditional probability...") == "theory"
        assert parser.classify_content_type("Solution: Let A be the event...") == "solution"

    def test_detect_mathematical_expressions(self, test_config):
        """Test mathematical expression detection."""
        parser = ContentParser(test_config)
        
        text = "Find P(A|B) where P(A) = 1/2 and P(B) = 1/3"
        expressions = parser.detect_mathematical_expressions(text)
        
        assert "P(A|B)" in expressions
        assert "1/2" in expressions
        assert "1/3" in expressions

    def test_identify_question_boundaries(self, test_config):
        """Test question boundary identification."""
        parser = ContentParser(test_config)
        
        text = """
        Exercise 30.3
        
        1. A die is thrown twice. Find the probability...
        
        2. In a class of 60 students...
        
        Solution to question 1: Let A be the event...
        """
        
        boundaries = parser.identify_question_boundaries(text)
        assert len(boundaries) >= 2
        assert all('question' in b['type'] for b in boundaries)

    def test_extract_question_numbering(self, test_config):
        """Test question numbering extraction."""
        parser = ContentParser(test_config)
        
        text = "1. First question. 2. Second question. Illustration 1: Example."
        
        numbering = parser.extract_question_numbering(text)
        assert "1" in numbering
        assert "2" in numbering
        assert "Illustration 1" in numbering

    def test_analyze_content_structure(self, test_config):
        """Test content structure analysis."""
        parser = ContentParser(test_config)
        
        text = """
        Chapter 30: Probability
        
        30.3 Conditional Probability
        
        Theory: Some theory here...
        
        Illustration 1: Example problem...
        
        Exercise 30.3
        
        1. Question one...
        2. Question two...
        """
        
        structure = parser.analyze_content_structure(text)
        assert structure['has_chapter_header']
        assert structure['has_exercises']
        assert structure['has_illustrations']

    def test_extract_content_relationships(self, test_config):
        """Test content relationship extraction."""
        parser = ContentParser(test_config)
        
        text = """
        Theory: Conditional probability is P(A|B) = P(A∩B)/P(B)
        
        Illustration 1: A bag contains 4 red balls...
        
        Exercise 30.3
        
        1. A die is thrown twice...
        """
        
        relationships = parser.extract_content_relationships(text)
        assert len(relationships) > 0
        assert any('theory' in rel['source'] for rel in relationships)

    def test_validate_content_quality(self, test_config):
        """Test content quality validation."""
        parser = ContentParser(test_config)
        
        # Test high quality content
        high_quality = "Find P(A|B) where A and B are independent events."
        quality = parser.validate_content_quality(high_quality)
        assert quality['score'] > 0.7
        
        # Test low quality content
        low_quality = "Find the thing with the stuff."
        quality = parser.validate_content_quality(low_quality)
        assert quality['score'] < 0.5

    def test_clean_and_normalize_text(self, test_config):
        """Test text cleaning and normalization."""
        parser = ContentParser(test_config)
        
        dirty_text = "  Find   P(A|B)   where   A   and   B   are   events.  "
        clean_text = parser.clean_and_normalize_text(dirty_text)
        
        assert clean_text == "Find P(A|B) where A and B are events."
        assert "  " not in clean_text  # No double spaces

    def test_extract_keywords(self, test_config):
        """Test keyword extraction."""
        parser = ContentParser(test_config)
        
        text = "Find the probability that both balls are red given that the first ball is red."
        keywords = parser.extract_keywords(text)
        
        assert "probability" in keywords
        assert "balls" in keywords
        assert "red" in keywords

    def test_detect_language(self, test_config):
        """Test language detection."""
        parser = ContentParser(test_config)
        
        english_text = "Find the probability of event A."
        hindi_text = "घटना A की प्रायिकता ज्ञात कीजिए।"
        
        assert parser.detect_language(english_text) == "en"
        # Note: This might need langdetect library for proper Hindi detection 