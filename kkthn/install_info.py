from __future__ import annotations

from pathlib import Path

from install_selector import selection_message


if __name__ == "__main__":
    print(selection_message(Path(__file__).resolve().parent))
