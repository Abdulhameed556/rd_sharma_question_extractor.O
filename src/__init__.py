"""
RD Sharma Question Extractor

A Retrieval-Augmented Generation (RAG) pipeline for extracting mathematical questions
from RD Sharma Class 12 textbook and formatting them in LaTeX.
"""

__version__ = "1.0.0"
__author__ = "RD Sharma Question Extractor Team"
__email__ = "support@rdsharma-extractor.com"

from .config import Config
from .main import QuestionExtractor

__all__ = ["Config", "QuestionExtractor"] 