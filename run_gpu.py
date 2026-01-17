#!/usr/bin/env python3
"""
Wrapper script to run HUNT GPU version from project root.
Adds src/ to path and runs main_gpu.py.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run
from main_gpu import main

if __name__ == '__main__':
    main()
