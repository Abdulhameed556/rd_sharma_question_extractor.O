"""
Notebook setup script - Run this at the beginning of each notebook
to ensure proper imports and environment setup.
"""

import sys
from pathlib import Path


def setup_notebook():
    """Setup the notebook environment for proper imports."""
    
    # Get the project root (two levels up from notebooks/)
    project_root = Path.cwd().parent
    src_path = project_root / "src"
    
    # Add src to Python path if not already there
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Test imports
    try:
        from main import QuestionExtractor  # noqa: F401
        from config import config  # noqa: F401
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False


# Run setup automatically when imported
if __name__ == "__main__":
    setup_notebook() 