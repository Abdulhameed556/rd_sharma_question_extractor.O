#!/usr/bin/env python3
"""
OCR quality testing script for RD Sharma Question Extractor.

This script tests OCR quality on sample pages and provides quality metrics
for mathematical content recognition.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_processor.ocr_processor import OCRProcessor
from document_processor.pdf_handler import PDFHandler
from config import config
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def test_ocr_quality(pdf_path: str, page_range: tuple = None, output_dir: str = None):
    """
    Test OCR quality on PDF pages.
    
    Args:
        pdf_path: Path to PDF file
        page_range: Tuple of (start_page, end_page) to test
        output_dir: Directory to save results
    """
    try:
        logger.info(f"Starting OCR quality test for: {pdf_path}")
        
        # Initialize components
        pdf_handler = PDFHandler(config)
        ocr_processor = OCRProcessor(config)
        
        # Load PDF
        pdf_handler.load_document(pdf_path)
        
        # Determine page range
        if page_range is None:
            start_page = 0
            end_page = min(5, pdf_handler.page_count - 1)  # Test first 5 pages
        else:
            start_page, end_page = page_range
        
        logger.info(f"Testing pages {start_page} to {end_page}")
        
        results = []
        
        for page_num in range(start_page, end_page + 1):
            logger.info(f"Processing page {page_num}")
            
            # Extract page image
            page_image = pdf_handler.extract_page_image(page_num)
            
            # Process with OCR
            ocr_result = ocr_processor.process_image(page_image)
            
            # Calculate quality metrics
            quality_metrics = calculate_ocr_quality(ocr_result, page_num)
            
            results.append({
                'page': page_num,
                'ocr_result': ocr_result,
                'quality_metrics': quality_metrics
            })
            
            logger.info(f"Page {page_num} - Confidence: {quality_metrics['confidence']:.2f}")
        
        # Generate summary
        summary = generate_quality_summary(results)
        
        # Save results if output directory specified
        if output_dir:
            save_ocr_results(results, summary, output_dir)
        
        # Print summary
        print_quality_summary(summary)
        
        return results, summary
        
    except Exception as e:
        logger.error(f"Error testing OCR quality: {e}")
        raise


def calculate_ocr_quality(ocr_result: dict, page_num: int) -> dict:
    """Calculate quality metrics for OCR result."""
    text = ocr_result.get('text', '')
    confidence = ocr_result.get('confidence', 0.0)
    
    # Basic metrics
    word_count = len(text.split())
    char_count = len(text)
    
    # Mathematical content detection
    math_expressions = detect_mathematical_content(text)
    math_density = len(math_expressions) / max(1, word_count)
    
    # Quality score based on confidence and content
    quality_score = confidence * (1 + math_density)
    
    return {
        'confidence': confidence,
        'word_count': word_count,
        'char_count': char_count,
        'math_expressions': math_expressions,
        'math_density': math_density,
        'quality_score': quality_score,
        'page': page_num
    }


def detect_mathematical_content(text: str) -> list:
    """Detect mathematical expressions in text."""
    import re
    
    math_patterns = [
        r'\d+/\d+',  # Fractions
        r'P\([^)]*\)',  # Probabilities
        r'[A-Z]\s*[∩∪⊂⊃∈∉]',  # Set operations
        r'[a-zA-Z]\s*=\s*[^=]+',  # Equations
        r'∑[^∑]*',  # Summations
        r'∫[^∫]*',  # Integrals
        r'√[^√]*',  # Roots
    ]
    
    expressions = []
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        expressions.extend(matches)
    
    return expressions


def generate_quality_summary(results: list) -> dict:
    """Generate quality summary from results."""
    if not results:
        return {}
    
    confidences = [r['quality_metrics']['confidence'] for r in results]
    quality_scores = [r['quality_metrics']['quality_score'] for r in results]
    math_densities = [r['quality_metrics']['math_density'] for r in results]
    
    return {
        'total_pages': len(results),
        'avg_confidence': sum(confidences) / len(confidences),
        'avg_quality_score': sum(quality_scores) / len(quality_scores),
        'avg_math_density': sum(math_densities) / len(math_densities),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences),
        'overall_quality': 'good' if sum(confidences) / len(confidences) > 0.8 else 'needs_improvement'
    }


def save_ocr_results(results: list, summary: dict, output_dir: str):
    """Save OCR test results to files."""
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / 'ocr_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    with open(output_path / 'ocr_quality_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def print_quality_summary(summary: dict):
    """Print quality summary to console."""
    print("\n" + "="*50)
    print("OCR QUALITY TEST SUMMARY")
    print("="*50)
    print(f"Total pages tested: {summary.get('total_pages', 0)}")
    print(f"Average confidence: {summary.get('avg_confidence', 0):.3f}")
    print(f"Average quality score: {summary.get('avg_quality_score', 0):.3f}")
    print(f"Average math density: {summary.get('avg_math_density', 0):.3f}")
    print(f"Confidence range: {summary.get('min_confidence', 0):.3f} - {summary.get('max_confidence', 0):.3f}")
    print(f"Overall quality: {summary.get('overall_quality', 'unknown')}")
    print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test OCR quality for RD Sharma Question Extractor"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to test"
    )
    parser.add_argument(
        "--pages",
        "-p",
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help="Page range to test (start end)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for results"
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
    
    # Test OCR quality
    try:
        test_ocr_quality(args.pdf_path, args.pages, args.output)
        print("✅ OCR quality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 