#!/usr/bin/env python3
"""
Setup script for RD Sharma Question Extractor.

This script automates the setup process for the RD Sharma question extractor,
including environment validation, dependency installation, and configuration setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any

def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment() -> bool:
    """Create a virtual environment."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment")

def install_dependencies() -> bool:
    """Install required dependencies."""
    # Determine the pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command([pip_cmd, "install", "--upgrade", "pip"], "Upgrading pip"):
        return False
    
    # Install requirements
    return run_command([pip_cmd, "install", "-r", "requirements.txt"], "Installing dependencies")

def create_directories() -> bool:
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data",
        "data/cache",
        "data/cache/ocr_cache",
        "data/cache/embeddings_cache",
        "outputs",
        "outputs/extracted_questions",
        "outputs/latex_files",
        "outputs/markdown_files",
        "outputs/logs",
        "notebooks",
        "tests",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully")
    return True

def setup_environment_file() -> bool:
    """Set up environment configuration file."""
    print("⚙️  Setting up environment configuration...")
    
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("❌ env.example file not found")
        return False
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Copy env.example to .env
    shutil.copy(env_example, env_file)
    print("✅ Created .env file from template")
    print("⚠️  Please edit .env file with your Groq API key")
    return True

def validate_groq_api_key() -> bool:
    """Validate Groq API key configuration."""
    print("🔑 Validating Groq API key...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read .env file
    api_key = None
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith("GROQ_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"\'')
                break
    
    if not api_key or api_key == "your_groq_api_key_here":
        print("⚠️  Groq API key not configured")
        print("   Please edit .env file and set GROQ_API_KEY to your actual API key")
        return False
    
    if len(api_key) < 20:
        print("❌ Groq API key appears to be invalid (too short)")
        return False
    
    print("✅ Groq API key appears to be configured")
    return True

def test_imports() -> bool:
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path("src")))
    
    try:
        import config
        print("✅ Config module imported successfully")
        
        from utils.logger import get_logger
        print("✅ Logger module imported successfully")
        
        from llm_interface.groq_client import GroqClient
        print("✅ Groq client module imported successfully")
        
        from main import QuestionExtractor
        print("✅ Main module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def run_tests() -> bool:
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Determine the python command based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/Mac
        python_cmd = "venv/bin/python"
    
    return run_command([python_cmd, "-m", "pytest", "tests/", "-v"], "Running tests")

def create_sample_data() -> bool:
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    # Create a sample PDF placeholder
    sample_pdf = Path("data/rd_sharma_complete.pdf")
    if not sample_pdf.exists():
        # Create a placeholder file
        with open(sample_pdf, 'w') as f:
            f.write("This is a placeholder for the RD Sharma PDF.\n")
            f.write("Please replace this file with the actual RD Sharma Class 12 PDF.\n")
        print("✅ Created placeholder PDF file")
    
    # Create sample document index
    sample_index = Path("data/document_index.json")
    if not sample_index.exists():
        import json
        index_data = {
            "document_info": {
                "title": "RD Sharma Class 12",
                "total_pages": 795,
                "chapters": list(range(1, 31))
            },
            "chapter_mappings": {
                "30": {
                    "start_page": 750,
                    "end_page": 795,
                    "topics": ["30.1", "30.2", "30.3", "30.4", "30.5", "30.6", "30.7", "30.8", "30.9"]
                }
            }
        }
        with open(sample_index, 'w') as f:
            json.dump(index_data, f, indent=2)
        print("✅ Created sample document index")
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up RD Sharma Question Extractor...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Setup environment file
    if not setup_environment_file():
        print("❌ Failed to setup environment file")
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("❌ Failed to create sample data")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Failed to import required modules")
        sys.exit(1)
    
    # Validate API key
    validate_groq_api_key()  # Don't exit on failure, just warn
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup completed")
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your Groq API key")
    print("2. Place RD Sharma PDF in data/rd_sharma_complete.pdf")
    print("3. Run: python src/main.py setup")
    print("4. Test extraction: python src/main.py extract 30 \"30.3\" --verbose")
    print("5. Open notebooks/demo.ipynb for interactive demo")
    
    print("\n🎯 Quick start commands:")
    print("   # Activate virtual environment")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    
    print("   # Run demo")
    print("   python src/main.py extract 30 \"30.3\" --output latex --verbose")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete setup and usage guide")
    print("   - notebooks/demo.ipynb: Interactive demonstration")
    print("   - src/main.py --help: Command-line help")

if __name__ == "__main__":
    main() 