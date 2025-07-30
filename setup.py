"""
Setup script for RD Sharma Question Extractor.

This package provides a complete RAG pipeline for extracting mathematical
questions from RD Sharma Class 12 textbook and formatting them in LaTeX.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="rd-sharma-question-extractor",
    version="1.0.0",
    author="RD Sharma Question Extractor Team",
    author_email="support@rdsharma-extractor.com",
    description="A RAG pipeline for extracting mathematical questions from RD Sharma Class 12 textbook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rd-sharma-question-extractor",
    project_urls={
        "Bug Reports": "https://github.com/your-username/rd-sharma-question-extractor/issues",
        "Source": "https://github.com/your-username/rd-sharma-question-extractor",
        "Documentation": "https://github.com/your-username/rd-sharma-question-extractor/blob/main/README.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup :: LaTeX",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rd-sharma-extract=main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords=[
        "mathematics",
        "education",
        "question-extraction",
        "latex",
        "rag",
        "llm",
        "ocr",
        "pdf-processing",
        "nlp",
        "machine-learning",
    ],
    license="MIT",
    platforms=["any"],
    zip_safe=False,
) 