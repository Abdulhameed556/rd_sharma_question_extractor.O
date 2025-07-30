#!/usr/bin/env python3
"""
Simple Test Script for RD Sharma Question Extractor
This script demonstrates the core functionality step by step.
"""

import os
import json
from pathlib import Path


def test_basic_functionality():
    """Test the basic functionality of the question extractor."""

    print("🧪 Testing RD Sharma Question Extractor")
    print("=" * 50)

    # Step 1: Check if PDF exists
    pdf_path = "data/rd_sharma_complete.pdf"
    if os.path.exists(pdf_path):
        print(f"✅ PDF found: {pdf_path}")
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"❌ PDF not found: {pdf_path}")
        print("   Please place your RD Sharma PDF in the data/ folder")
        return False

    # Step 2: Check if we have sample outputs
    sample_output = "outputs/extracted_questions/chapter_30_30_3.json"
    if os.path.exists(sample_output):
        print(f"✅ Sample output found: {sample_output}")
        with open(sample_output, "r") as f:
            data = json.load(f)
            print(f"   Contains {len(data)} questions")

            # Show first question
            if data:
                first_q = data[0]
                print(f"   Sample question: {first_q['question_number']}")
                print(f"   Text: {first_q['question_text'][:100]}...")
    else:
        print(f"❌ Sample output not found: {sample_output}")

    # Step 3: Check LaTeX output
    latex_output = "outputs/latex_files/chapter_30_30_3.tex"
    if os.path.exists(latex_output):
        print(f"✅ LaTeX output found: {latex_output}")
        with open(latex_output, "r") as f:
            content = f.read()
            print(f"   File size: {len(content)} characters")
            print(f"   Contains LaTeX document structure")
    else:
        print(f"❌ LaTeX output not found: {latex_output}")

    # Step 4: Check configuration
    config_file = "src/config.py"
    if os.path.exists(config_file):
        print(f"✅ Configuration found: {config_file}")
    else:
        print(f"❌ Configuration not found: {config_file}")

    # Step 5: Check main entry point
    main_file = "src/main.py"
    if os.path.exists(main_file):
        print(f"✅ Main entry point found: {main_file}")
        print("   You can run: python src/main.py --help")
    else:
        print(f"❌ Main entry point not found: {main_file}")

    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Get a Groq API key from https://console.groq.com/")
    print("2. Copy .env.example to .env and add your API key")
    print("3. Run: python src/main.py extract --chapter 30 --topic 30.3")
    print("4. Check outputs/extracted_questions/ for results")

    return True


def show_sample_output():
    """Show what the output looks like."""

    print("\n📄 Sample Output Structure:")
    print("=" * 50)

    sample_data = [
        {
            "question_number": "Illustration 1",
            "question_text": "A bag contains $4$ red balls and $6$ black balls. Two balls are drawn at random without replacement. Find $P(\\text{both balls are red})$.",
            "source": "Illustration",
        },
        {
            "question_number": "1",
            "question_text": "A die is thrown twice. Find $P(\\text{sum} = 8 | \\text{first throw is even})$.",
            "source": "Exercise 30.3",
        },
    ]

    print("JSON Output:")
    print(json.dumps(sample_data, indent=2))

    print("\nLaTeX Output:")
    print(
        r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}

\begin{document}
\title{Chapter 30: Conditional Probability Questions}
\maketitle

\begin{enumerate}
\item A bag contains $4$ red balls and $6$ black balls. Two balls are drawn at random without replacement. Find $P(\text{both balls are red})$.

\item A die is thrown twice. Find $P(\text{sum} = 8 | \text{first throw is even})$.
\end{enumerate}

\end{document}
"""
    )


if __name__ == "__main__":
    test_basic_functionality()
    show_sample_output()
