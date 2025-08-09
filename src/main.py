"""
Main CLI interface for RD Sharma Question Extractor.

This module provides the command-line interface and main application logic
for extracting mathematical questions from RD Sharma textbook.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .config import config
from .utils.logger import get_logger, log_exception, log_extraction_start, log_extraction_complete
from .utils.exceptions import BaseExtractorError
from .llm_interface.groq_client import GroqClient

# Initialize Typer app and Rich console
app = typer.Typer(help="RD Sharma Question Extractor - Extract mathematical questions in LaTeX format")
console = Console()
logger = get_logger(__name__)


class QuestionExtractor:
    """Main question extraction orchestrator."""
    
    def __init__(self):
        """Initialize the question extractor with all components."""
        self.groq_client = GroqClient()
        self.console = console
        
        # Create necessary directories
        config.create_directories()
        
        logger.info("QuestionExtractor initialized successfully")
    
    def extract_questions(
        self, 
        chapter: int, 
        topic: str, 
        output_format: str = "json",
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract questions from a specific chapter and topic.
        
        Args:
            chapter: Chapter number
            topic: Topic identifier (e.g., "30.3")
            output_format: Output format ("json", "latex", "markdown")
            verbose: Enable verbose output
            
        Returns:
            List of extracted questions
        """
        start_time = time.time()
        
        try:
            log_extraction_start(logger, chapter, topic)
            
            # For now, we'll use a sample content since we don't have the actual PDF
            # In a real implementation, this would load from the PDF
            sample_content = self._get_sample_content(chapter, topic)
            
            if verbose:
                self.console.print(f"[bold blue]Processing Chapter {chapter}, Topic {topic}[/bold blue]")
                self.console.print(f"[dim]Content length: {len(sample_content)} characters[/dim]")
            
            # Extract questions using LLM
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Extracting questions...", total=None)
                
                questions = self.groq_client.extract_questions(sample_content, chapter, topic)
                
                progress.update(task, description="✅ Questions extracted successfully")
            
            # Validate results
            validation_results = self.groq_client.validate_response(questions)
            
            if verbose:
                self._display_validation_results(validation_results)
            
            # Format output
            if output_format == "latex":
                return self._format_latex_output(questions, chapter, topic)
            elif output_format == "markdown":
                return self._format_markdown_output(questions, chapter, topic)
            else:
                return questions
            
        except Exception as e:
            log_exception(logger, e, {"chapter": chapter, "topic": topic})
            raise
        finally:
            duration = time.time() - start_time
            log_extraction_complete(logger, chapter, topic, len(questions) if 'questions' in locals() else 0, duration)
    
    def _get_sample_content(self, chapter: int, topic: str) -> str:
        """
        Get sample content for testing purposes.
        In a real implementation, this would load from the actual PDF.
        """
        if chapter == 30 and topic == "30.3":
            return """Chapter 30: Probability

30.3 Conditional Probability

Theory: Conditional probability is defined as P(A|B) = P(A∩B)/P(B), where P(B) > 0.

Illustration 1: A bag contains 4 red balls and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both balls are red.

More theory about conditional probability and its applications...

Exercise 30.3

1. A die is thrown twice. Find the probability that the sum of numbers appearing is 8, given that the first throw shows an even number.

2. In a class of 60 students, 30 play cricket, 20 play football and 10 play both games. A student is selected at random. Find the probability that:
   (i) He plays cricket given that he plays football
   (ii) He plays exactly one game

Solution to question 1: Let A be the event that sum is 8, B be the event that first throw is even...

3. Two cards are drawn successively without replacement from a well-shuffled pack of 52 cards. Find the probability that both cards are aces given that at least one is an ace.

Additional theory about Bayes' theorem and its applications in probability...

4. A box contains 3 red balls, 4 white balls and 5 blue balls. Three balls are drawn at random without replacement. Find the probability that all three balls are of different colors."""
        else:
            return f"Sample content for Chapter {chapter}, Topic {topic}. This is placeholder content for testing purposes."
    
    def _display_validation_results(self, results: Dict[str, Any]):
        """Display validation results in a formatted table."""
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Questions", str(results["total_questions"]))
        table.add_row("Valid Questions", str(results["valid_questions"]))
        table.add_row("Quality Score", f"{results['quality_score']:.2%}")
        
        if results["latex_errors"]:
            table.add_row("LaTeX Errors", str(len(results["latex_errors"])))
        
        if results["missing_fields"]:
            table.add_row("Missing Fields", str(len(results["missing_fields"])))
        
        self.console.print(table)
    
    def _format_latex_output(self, questions: List[Dict[str, Any]], chapter: int, topic: str) -> str:
        """Format questions as LaTeX document."""
        latex_content = f"""\\section{{Chapter {chapter}: {topic} Questions}}

\\begin{{enumerate}}
"""
        
        for question in questions:
            latex_content += f"\\item {question['question_text']}\n"
        
        latex_content += "\\end{enumerate}"
        
        return latex_content
    
    def _format_markdown_output(self, questions: List[Dict[str, Any]], chapter: int, topic: str) -> str:
        """Format questions as Markdown document."""
        markdown_content = f"# Chapter {chapter}: {topic} Questions\n\n"
        
        for i, question in enumerate(questions, 1):
            markdown_content += f"## Question {i}\n\n{question['question_text']}\n\n"
        
        return markdown_content
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the document structure."""
        return {
            "total_pages": 795,  # RD Sharma Class 12
            "chapters": list(range(1, 31)),  # Chapters 1-30
            "document_path": config.pdf_path,
            "index_path": config.document_index_path
        }


@app.command()
def extract(
    chapter: int = typer.Argument(..., help="Chapter number (1-30)"),
    topic: str = typer.Argument(..., help="Topic identifier (e.g., '30.3')"),
    output: str = typer.Option("json", "--output", "-o", help="Output format: json, latex, markdown"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save output to file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """
    Extract mathematical questions from RD Sharma textbook.
    
    Example:
        python src/main.py extract 30 "30.3" --output latex --verbose
    """
    try:
        # Validate inputs
        if not 1 <= chapter <= 30:
            console.print(f"[red]Error: Chapter must be between 1 and 30, got {chapter}[/red]")
            raise typer.Exit(1)
        
        if not topic:
            console.print("[red]Error: Topic cannot be empty[/red]")
            raise typer.Exit(1)
        
        # Initialize extractor
        extractor = QuestionExtractor()
        
        # Display header
        console.print(Panel.fit(
            f"[bold blue]RD Sharma Question Extractor[/bold blue]\n"
            f"Chapter {chapter} • Topic {topic} • Output: {output.upper()}",
            border_style="blue"
        ))
        
        # Extract questions
        result = extractor.extract_questions(chapter, topic, output, verbose)
        
        # Display results
        if output == "json":
            console.print("\n[bold green]Extracted Questions:[/bold green]")
            for i, question in enumerate(result, 1):
                console.print(f"\n[cyan]Question {i}:[/cyan]")
                console.print(f"  Number: {question.get('question_number', 'N/A')}")
                console.print(f"  Source: {question.get('source', 'N/A')}")
                console.print(f"  Text: {question.get('question_text', 'N/A')}")
        else:
            console.print("\n[bold green]Generated Output:[/bold green]")
            console.print(Panel(result, border_style="green"))
        
        # Save to file if requested
        if save:
            output_file = f"outputs/chapter_{chapter}_{topic.replace('.', '_')}.{output}"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if output == "json":
                    import json
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    f.write(result)
            
            console.print(f"\n[green]✅ Output saved to: {output_file}[/green]")
        
        console.print(f"\n[bold green]✅ Extraction completed successfully![/bold green]")
        
    except BaseExtractorError as e:
        console.print(f"\n[red]❌ Extraction Error: {e}[/red]")
        if debug:
            console.print(f"[dim]Error details: {e.to_dict()}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected Error: {e}[/red]")
        if debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display information about the document and available chapters."""
    try:
        extractor = QuestionExtractor()
        doc_info = extractor.get_document_info()
        
        console.print(Panel.fit(
            "[bold blue]RD Sharma Class 12 Textbook Information[/bold blue]",
            border_style="blue"
        ))
        
        table = Table(title="Document Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Pages", str(doc_info["total_pages"]))
        table.add_row("Total Chapters", str(len(doc_info["chapters"])))
        table.add_row("Document Path", doc_info["document_path"])
        table.add_row("Index Path", doc_info["index_path"])
        
        console.print(table)
        
        console.print("\n[bold green]Available Chapters:[/bold green]")
        chapters_text = ", ".join(str(ch) for ch in doc_info["chapters"])
        console.print(Panel(chapters_text, border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error getting document info: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    chapter: int = typer.Argument(..., help="Chapter number to validate"),
    topic: str = typer.Argument(..., help="Topic to validate")
):
    """Validate extraction quality for a specific chapter and topic."""
    try:
        extractor = QuestionExtractor()
        
        console.print(f"[bold blue]Validating Chapter {chapter}, Topic {topic}[/bold blue]")
        
        # Extract questions
        questions = extractor.extract_questions(chapter, topic, "json", verbose=True)
        
        # Validate
        validation_results = extractor.groq_client.validate_response(questions)
        
        # Display detailed validation
        console.print("\n[bold green]Detailed Validation Results:[/bold green]")
        
        if validation_results["latex_errors"]:
            console.print("\n[red]LaTeX Errors:[/red]")
            for error in validation_results["latex_errors"]:
                console.print(f"  • {error}")
        
        if validation_results["missing_fields"]:
            console.print("\n[red]Missing Fields:[/red]")
            for field in validation_results["missing_fields"]:
                console.print(f"  • {field}")
        
        quality_score = validation_results["quality_score"]
        if quality_score >= 0.9:
            console.print(f"\n[green]✅ Excellent Quality Score: {quality_score:.2%}[/green]")
        elif quality_score >= 0.7:
            console.print(f"\n[yellow]⚠️  Good Quality Score: {quality_score:.2%}[/yellow]")
        else:
            console.print(f"\n[red]❌ Poor Quality Score: {quality_score:.2%}[/red]")
        
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def setup():
    """Set up the environment and validate configuration."""
    try:
        console.print("[bold blue]Setting up RD Sharma Question Extractor...[/bold blue]")
        
        # Validate configuration
        console.print("\n[cyan]Validating configuration...[/cyan]")
        
        # Check API key
        if config.groq_api_key and config.groq_api_key != "your_groq_api_key_here":
            console.print("✅ Groq API key configured")
        else:
            console.print("❌ Groq API key not configured")
            console.print("Please set GROQ_API_KEY in your .env file")
            raise typer.Exit(1)
        
        # Check PDF path
        pdf_path = Path(config.pdf_path)
        if pdf_path.exists():
            console.print(f"✅ PDF found: {config.pdf_path}")
        else:
            console.print(f"⚠️  PDF not found: {config.pdf_path}")
            console.print("Please ensure the RD Sharma PDF is in the data/ directory")
        
        # Create directories
        console.print("\n[cyan]Creating directories...[/cyan]")
        config.create_directories()
        console.print("✅ Directories created")
        
        # Test Groq connection
        console.print("\n[cyan]Testing Groq API connection...[/cyan]")
        try:
            groq_client = GroqClient()
            console.print("✅ Groq API connection successful")
        except Exception as e:
            console.print(f"❌ Groq API connection failed: {e}")
            raise typer.Exit(1)
        
        console.print("\n[bold green]✅ Setup completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Setup error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 