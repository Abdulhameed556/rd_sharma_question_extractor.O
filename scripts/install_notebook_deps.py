#!/usr/bin/env python3
"""
Install notebook dependencies and check Python environment.
Run this script to ensure all required packages are available.
"""

import sys
import subprocess
import importlib


def check_and_install_package(package_name, pip_name=None):
    """Check if a package is installed, install if missing."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is missing, installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name
            ])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False


def main():
    """Install all required notebook dependencies."""
    print("🔧 Installing notebook dependencies...")
    print("=" * 50)
    
    # List of required packages
    packages = [
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("psutil", "psutil"),
        ("plotly", "plotly"),
        ("bokeh", "bokeh"),
        ("scipy", "scipy"),
        ("numpy", "numpy")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, pip_name in packages:
        if check_and_install_package(package_name, pip_name):
            success_count += 1
    
    print("\n📊 Installation Summary:")
    print(f"   ✅ Successfully installed: {success_count}/{total_count}")
    print(f"   ❌ Failed installations: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        print("💡 You can now run the full notebooks with visualizations.")
    else:
        print("⚠️  Some dependencies failed to install.")
        print("💡 You can still use the simplified demo notebook.")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import pandas as pd  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401
        print("✅ All visualization packages imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    main() 