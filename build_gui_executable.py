#!/usr/bin/env python
"""
Build script: Package Grayspot GUI as executable using PyInstaller.

Generates standalone .exe (Windows) / .app (macOS) / binary (Linux)
without requiring Python installation.

Usage:
    python build_gui_executable.py
    
Output:
    dist/grayspot/grayspot.exe  (Windows)
    dist/grayspot/grayspot      (Linux)
    dist/Grayspot.app           (macOS)
"""

import subprocess
import sys
from pathlib import Path

def check_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True

def build_executable():
    """Build GUI executable using PyInstaller."""
    root = Path(__file__).parent
    
    print("🔨 Building Grayspot GUI Executable...")
    print(f"   Root: {root}")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",                    # Create single directory
        "--windowed",                  # GUI mode (no console)
        "--name", "grayspot",
        "--icon", str(root / "gui/assets/icon.ico"),  # Optional icon
        "--add-data", f"{root}/src:src",
        "--add-data", f"{root}/gui:gui",
        str(root / "gui/app.py"),
    ]
    
    print(f"\n📦 Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Build successful!")
        print(f"   Output: {root}/dist/grayspot/")
        print(f"\n💡 To run the executable:")
        print(f"   - Windows: dist/grayspot/grayspot.exe")
        print(f"   - Linux:   dist/grayspot/grayspot")
        print(f"   - macOS:   dist/Grayspot.app")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return False

if __name__ == "__main__":
    if check_pyinstaller():
        success = build_executable()
        sys.exit(0 if success else 1)
