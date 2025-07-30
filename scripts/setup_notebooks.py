#!/usr/bin/env python3
"""
Setup script for Jupyter notebooks to ensure proper module imports.
This script configures the Python path and creates necessary directories.
"""

import sys
from pathlib import Path
import subprocess


def setup_notebook_environment():
    """Setup the environment for Jupyter notebooks."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    print("🔧 Setting up notebook environment...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")
    
    # Add src to Python path
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Create necessary directories
    directories = [
        project_root / "outputs",
        project_root / "data" / "cache",
        project_root / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install the project in development mode if not already installed
    try:
        import src  # noqa: F401
        print("✅ Project modules are accessible")
    except ImportError:
        print("⚠️  Installing project in development mode...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=project_root)
        print("✅ Project installed in development mode")
    
    # Test imports
    try:
        from src.main import QuestionExtractor  # noqa: F401
        from src.config import config  # noqa: F401
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    setup_notebook_environment() 