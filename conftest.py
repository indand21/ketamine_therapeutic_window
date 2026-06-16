"""Pytest configuration: ensure the repository root is importable.

Allows ``import src...`` from the test suite without requiring an editable
install during the WP1 implementation phase.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
