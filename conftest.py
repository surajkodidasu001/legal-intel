"""Auto-loaded by pytest before test collection. Puts src/ on sys.path so
`import legalintel...` works in test files without needing PYTHONPATH set
manually or an installed package -- same approach scripts/prepare_data.py
and scripts/profile_dedup.py already use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
