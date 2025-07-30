"""
Document indexer for RD Sharma Question Extractor.

This module handles document structure analysis, chapter/topic boundary detection,
and page-to-content mapping for efficient retrieval.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.exceptions import DocumentProcessingError
from ..config import config

logger = get_logger(__name__)


@dataclass
class ChapterInfo:
    """Chapter information."""
    chapter_number: str
    title: str
    start_page: int
    end_page: int
    topics: List[Dict[str, Any]]


@dataclass
class TopicInfo:
    """Topic information."""
    topic_number: str
    title: str
    start_page: int
    end_page: int
    chapter: str
    content_types: List[str]


class DocumentIndexer:
    """Handles document indexing and structure analysis."""

    def __init__(self, config):
        """Initialize document indexer."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Document structure
        self.document_index = {}
        self.chapters = []
        self.topics = []
        self.page_mapping = {}
        
        # Index file path
        self.index_path = Path(config.document_index_path)
        
        # Pattern matching
        self.chapter_patterns = [
            r'Chapter\s+(\d+):\s*(.+)',
            r'CHAPTER\s+(\d+):\s*(.+)',
            r'(\d+)\.\s*(.+)',  # Simple numbered format
        ]
        
        self.topic_patterns = [
            r'(\d+\.\d+)\s+(.+)',  # 30.1 Introduction
            r'(\d+\.\d+):\s*(.+)',  # 30.1: Introduction
            r'(\d+\.\d+)\s*[-–]\s*(.+)',  # 30.1 - Introduction
        ]
        
        self.content_type_patterns = {
            'illustration': r'Illustration\s+\d+',
            'exercise': r'Exercise\s+\d+\.\d+',
            'theory': r'Theory|Definition|Theorem|Lemma|Corollary',
            'solution': r'Solution|Answer',
            'example': r'Example\s+\d+',
        }

    def build_document_index(self, pdf_path: str) -> Dict[str, Any]:
        """
        Build comprehensive document index from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document index dictionary
        """
        try:
            self.logger.info(f"Building document index for: {pdf_path}")
            
            # Extract all pages
            pages = self._extract_all_pages(pdf_path)
            
            # Extract table of contents
            toc = self.extract_table_of_contents(pages)
            
            # Detect chapter boundaries
            chapters = self.detect_chapter_boundaries(pages)
            
            # Detect topic boundaries within chapters
            topics = self.detect_topic_boundaries(pages, chapters)
            
            # Create page mapping
            page_mapping = self.create_page_mapping(chapters, topics)
            
            # Build comprehensive index
            document_index = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'pdf_path': pdf_path,
                    'total_pages': len(pages),
                    'total_chapters': len(chapters),
                    'total_topics': len(topics),
                    'version': '1.0'
                },
                'table_of_contents': toc,
                'chapters': chapters,
                'topics': topics,
                'page_mapping': page_mapping
            }
            
            self.document_index = document_index
            self.chapters = chapters
            self.topics = topics
            self.page_mapping = page_mapping
            
            # Save index
            self.save_index(document_index)
            
            self.logger.info(f"Document index built successfully: {len(chapters)} chapters, {len(topics)} topics")
            return document_index
            
        except Exception as e:
            self.logger.error(f"Error building document index: {e}")
            raise DocumentProcessingError(f"Failed to build document index: {str(e)}")

    def extract_table_of_contents(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract table of contents from document pages.
        
        Args:
            pages: List of page dictionaries with content
            
        Returns:
            List of TOC entries
        """
        toc_entries = []
        
        for page in pages:
            content = page.get('content', '')
            page_num = page.get('page', 0)
            
            # Look for TOC patterns
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Match chapter patterns
                for pattern in self.chapter_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        chapter_num = match.group(1)
                        title = match.group(2).strip()
                        
                        toc_entries.append({
                            'type': 'chapter',
                            'number': chapter_num,
                            'title': title,
                            'page': page_num,
                            'line': line
                        })
                        break
                
                # Match topic patterns
                for pattern in self.topic_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        topic_num = match.group(1)
                        title = match.group(2).strip()
                        
                        # Extract chapter number from topic
                        chapter_num = topic_num.split('.')[0]
                        
                        toc_entries.append({
                            'type': 'topic',
                            'number': topic_num,
                            'title': title,
                            'chapter': chapter_num,
                            'page': page_num,
                            'line': line
                        })
                        break
        
        return toc_entries

    def detect_chapter_boundaries(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect chapter boundaries in document.
        
        Args:
            pages: List of page dictionaries with content
            
        Returns:
            List of chapter information
        """
        chapters = []
        current_chapter = None
        
        for i, page in enumerate(pages):
            content = page.get('content', '')
            page_num = page.get('page', i)
            
            # Look for chapter headers
            for pattern in self.chapter_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    chapter_num = match.group(1)
                    title = match.group(2).strip()
                    
                    # Close previous chapter if exists
                    if current_chapter:
                        current_chapter['end_page'] = page_num - 1
                        chapters.append(current_chapter)
                    
                    # Start new chapter
                    current_chapter = {
                        'chapter': chapter_num,
                        'title': title,
                        'start_page': page_num,
                        'end_page': None,
                        'page_count': 0
                    }
                    break
        
        # Close last chapter
        if current_chapter:
            current_chapter['end_page'] = len(pages) - 1
            chapters.append(current_chapter)
        
        # Calculate page counts
        for chapter in chapters:
            chapter['page_count'] = chapter['end_page'] - chapter['start_page'] + 1
        
        return chapters

    def detect_topic_boundaries(self, pages: List[Dict[str, Any]], 
                              chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect topic boundaries within chapters.
        
        Args:
            pages: List of page dictionaries with content
            chapters: List of chapter information
            
        Returns:
            List of topic information
        """
        topics = []
        
        for chapter in chapters:
            chapter_num = chapter['chapter']
            start_page = chapter['start_page']
            end_page = chapter['end_page']
            
            chapter_topics = []
            current_topic = None
            
            # Process pages within chapter
            for page_num in range(start_page, end_page + 1):
                if page_num >= len(pages):
                    break
                
                page = pages[page_num]
                content = page.get('content', '')
                
                # Look for topic headers
                for pattern in self.topic_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        topic_num = match.group(1)
                        title = match.group(2).strip()
                        
                        # Check if topic belongs to this chapter
                        if topic_num.startswith(f"{chapter_num}."):
                            # Close previous topic if exists
                            if current_topic:
                                current_topic['end_page'] = page_num - 1
                                chapter_topics.append(current_topic)
                            
                            # Start new topic
                            current_topic = {
                                'topic': topic_num,
                                'title': title,
                                'chapter': chapter_num,
                                'start_page': page_num,
                                'end_page': None,
                                'content_types': []
                            }
                            break
            
            # Close last topic in chapter
            if current_topic:
                current_topic['end_page'] = end_page
                chapter_topics.append(current_topic)
            
            # Analyze content types for each topic
            for topic in chapter_topics:
                topic['content_types'] = self._analyze_topic_content_types(
                    pages, topic['start_page'], topic['end_page']
                )
            
            topics.extend(chapter_topics)
        
        return topics

    def create_page_mapping(self, chapters: List[Dict[str, Any]], 
                           topics: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Create page-to-content mapping.
        
        Args:
            chapters: List of chapter information
            topics: List of topic information
            
        Returns:
            Dictionary mapping page numbers to content information
        """
        page_mapping = {}
        
        # Map chapters to pages
        for chapter in chapters:
            for page_num in range(chapter['start_page'], chapter['end_page'] + 1):
                page_mapping[page_num] = {
                    'chapter': chapter['chapter'],
                    'chapter_title': chapter['title'],
                    'topic': None,
                    'topic_title': None,
                    'content_types': []
                }
        
        # Map topics to pages
        for topic in topics:
            for page_num in range(topic['start_page'], topic['end_page'] + 1):
                if page_num in page_mapping:
                    page_mapping[page_num].update({
                        'topic': topic['topic'],
                        'topic_title': topic['title'],
                        'content_types': topic['content_types']
                    })
                else:
                    # Page not in chapter mapping (shouldn't happen)
                    page_mapping[page_num] = {
                        'chapter': topic['chapter'],
                        'chapter_title': f"Chapter {topic['chapter']}",
                        'topic': topic['topic'],
                        'topic_title': topic['title'],
                        'content_types': topic['content_types']
                    }
        
        return page_mapping

    def _analyze_topic_content_types(self, pages: List[Dict[str, Any]], 
                                   start_page: int, end_page: int) -> List[str]:
        """Analyze content types within a topic."""
        content_types = set()
        
        for page_num in range(start_page, end_page + 1):
            if page_num >= len(pages):
                break
            
            content = pages[page_num].get('content', '')
            
            # Check for content type patterns
            for content_type, pattern in self.content_type_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    content_types.add(content_type)
        
        return list(content_types)

    def _extract_all_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract all pages from PDF."""
        # This would integrate with PDFHandler
        # For now, return empty list as placeholder
        return []

    def get_chapter_pages(self, chapter: str) -> Optional[Dict[str, Any]]:
        """
        Get page range for a specific chapter.
        
        Args:
            chapter: Chapter number
            
        Returns:
            Chapter information or None if not found
        """
        for chapter_info in self.chapters:
            if chapter_info['chapter'] == chapter:
                return chapter_info
        return None

    def get_topic_pages(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get page range for a specific topic.
        
        Args:
            topic: Topic number (e.g., "30.3")
            
        Returns:
            Topic information or None if not found
        """
        for topic_info in self.topics:
            if topic_info['topic'] == topic:
                return topic_info
        return None

    def get_chapter_topics(self, chapter: str) -> List[Dict[str, Any]]:
        """
        Get all topics within a chapter.
        
        Args:
            chapter: Chapter number
            
        Returns:
            List of topic information
        """
        return [topic for topic in self.topics if topic['chapter'] == chapter]

    def search_content(self, query: str, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for content matching query.
        
        Args:
            query: Search query
            content_type: Optional content type filter
            
        Returns:
            List of matching content
        """
        results = []
        
        for topic in self.topics:
            # Check if topic matches content type filter
            if content_type and content_type not in topic['content_types']:
                continue
            
            # Check if topic title matches query
            if query.lower() in topic['title'].lower():
                results.append({
                    'type': 'topic',
                    'topic': topic['topic'],
                    'title': topic['title'],
                    'chapter': topic['chapter'],
                    'start_page': topic['start_page'],
                    'end_page': topic['end_page']
                })
        
        return results

    def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        if not self.document_index:
            return {}
        
        stats = {
            'total_chapters': len(self.chapters),
            'total_topics': len(self.topics),
            'total_pages': len(self.page_mapping),
            'chapters': {}
        }
        
        # Chapter statistics
        for chapter in self.chapters:
            chapter_topics = self.get_chapter_topics(chapter['chapter'])
            stats['chapters'][chapter['chapter']] = {
                'title': chapter['title'],
                'page_count': chapter['page_count'],
                'topic_count': len(chapter_topics),
                'start_page': chapter['start_page'],
                'end_page': chapter['end_page']
            }
        
        return stats

    def save_index(self, index_data: Dict[str, Any]):
        """Save document index to file."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Document index saved to: {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving document index: {e}")
            raise DocumentProcessingError(f"Failed to save document index: {str(e)}")

    def load_index(self) -> Dict[str, Any]:
        """Load document index from file."""
        try:
            if not self.index_path.exists():
                self.logger.warning(f"Index file not found: {self.index_path}")
                return {}
            
            with open(self.index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Restore object state
            self.document_index = index_data
            self.chapters = index_data.get('chapters', [])
            self.topics = index_data.get('topics', [])
            self.page_mapping = index_data.get('page_mapping', {})
            
            self.logger.info(f"Document index loaded from: {self.index_path}")
            return index_data
            
        except Exception as e:
            self.logger.error(f"Error loading document index: {e}")
            raise DocumentProcessingError(f"Failed to load document index: {str(e)}")

    def update_index(self, pdf_path: str):
        """Update document index."""
        self.logger.info(f"Updating document index for: {pdf_path}")
        
        # Check if PDF has changed
        if self._has_pdf_changed(pdf_path):
            self.build_document_index(pdf_path)
        else:
            self.logger.info("PDF unchanged, using existing index")

    def _has_pdf_changed(self, pdf_path: str) -> bool:
        """Check if PDF file has changed since last indexing."""
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            return True
        
        if not self.index_path.exists():
            return True
        
        # Compare modification times
        pdf_mtime = pdf_path_obj.stat().st_mtime
        index_mtime = self.index_path.stat().st_mtime
        
        return pdf_mtime > index_mtime

    def export_index_summary(self, output_path: str):
        """Export index summary to file."""
        try:
            summary = {
                'metadata': self.document_index.get('metadata', {}),
                'statistics': self.get_document_statistics(),
                'chapters': [
                    {
                        'chapter': ch['chapter'],
                        'title': ch['title'],
                        'topics': [
                            {
                                'topic': t['topic'],
                                'title': t['title'],
                                'content_types': t['content_types']
                            }
                            for t in self.get_chapter_topics(ch['chapter'])
                        ]
                    }
                    for ch in self.chapters
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Index summary exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting index summary: {e}")
            raise DocumentProcessingError(f"Failed to export index summary: {str(e)}") 