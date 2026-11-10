"""Microstructure & Fabric Explorer CLI. Run with --help for options."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demviz import main

if __name__ == "__main__":
    main("spherical")
