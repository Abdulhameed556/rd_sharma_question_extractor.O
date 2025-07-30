#!/usr/bin/env python3
"""
Build document index script for RD Sharma Question Extractor.

This script performs one-time document indexing to create a comprehensive
index of chapters, topics, and page mappings for efficient retrieval.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_processor.document_indexer import DocumentIndexer
from config import config
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def build_document_index(pdf_path: str, output_path: str = None):
    """
    Build comprehensive document index.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path for index
    """
    try:
        logger.info(f"Starting document index build for: {pdf_path}")
        
        # Initialize document indexer
        indexer = DocumentIndexer(config)
        
        # Build index
        document_index = indexer.build_document_index(pdf_path)
        
        # Save to custom path if specified
        if output_path:
            indexer.save_index(document_index)
            logger.info(f"Index saved to: {output_path}")
        
        # Print statistics
        stats = indexer.get_document_statistics()
        logger.info("Document indexing completed successfully!")
        logger.info(f"Total chapters: {stats.get('total_chapters', 0)}")
        logger.info(f"Total topics: {stats.get('total_topics', 0)}")
        logger.info(f"Total pages: {stats.get('total_pages', 0)}")
        
        return document_index
        
    except Exception as e:
        logger.error(f"Error building document index: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build document index for RD Sharma Question Extractor"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to index"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the index file (optional)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    
    # Build index
    try:
        build_document_index(args.pdf_path, args.output)
        print("✅ Document index built successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 