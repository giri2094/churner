"""Shared pytest configuration for the ``churner`` test suite.

The tests live in ``<project_root>/tests`` and the source code lives under
``<project_root>/src``. Adding that directory to ``sys.path`` lets the tests
import the package without it having been installed, which is the same approach
the scripts under ``<project_root>/scripts`` use.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))
