import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buggy import add  # noqa: E402


def test_add() -> None:
    assert add(2, 1) == 3
