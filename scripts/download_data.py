"""Download a training corpus into data/input.txt.

Boilerplate: fetches Karpathy's 'tinyshakespeare' (~1.1 MB of Shakespeare,
concatenated) — the standard tiny corpus for character-level language models.
Run with:  uv run python scripts/download_data.py
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST} ({DEST.stat().st_size:,} bytes)")
        return
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
