"""Character-level tokenizer.

A tokenizer does exactly two things:
  - encode: text (str)  -> list of integer IDs
  - decode: list of IDs -> text (str)

For character-level, each unique character in the corpus is one token. The
vocabulary is the *sorted* set of unique characters, and a character's ID is its
index in that sorted list (sorted so the mapping is deterministic / reproducible
across runs).

You implement the three TODOs. Then run:
    uv run python -m src.tokenizer
to exercise the smoke test at the bottom.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, text: str) -> None:
        # The vocabulary: every distinct character in `text`, in sorted order.
        # set(text) drops duplicates; sorted(...) fixes a deterministic order.
        self.chars = sorted(set(text))                # e.g. ['\n', ' ', '!', ...]

        # How many distinct tokens exist. Later this is the length of every
        # probability vector the model outputs.
        self.vocab_size = len(self.chars)             # an int (65 for tinyshakespeare)

        # Two lookup tables built from the same (id, char) pairs:
        #   stoi: "string-to-integer", char -> ID   (used by encode)
        #   itos: "integer-to-string", ID -> char   (used by decode)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # {char: id}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # {id: char}

    def encode(self, s: str) -> list[int]:
        """Turn a string into a list of token IDs."""
        return [self.stoi[c] for c in s]              # look up each character

    def decode(self, ids: list[int]) -> str:
        """Turn a list of token IDs back into a string."""
        return "".join(self.itos[i] for i in ids)     # look up each id, then glue


if __name__ == "__main__":
    # --- smoke test (boilerplate; you don't need to edit this) ---
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "data" / "input.txt").read_text(
        encoding="utf-8"
    )
    tok = CharTokenizer(text)

    print(f"vocab_size = {tok.vocab_size}")
    print(f"first 20 chars of vocab: {tok.chars[:20]}")

    sample = "First Citizen:\nWe are accounted poor."
    ids = tok.encode(sample)
    back = tok.decode(ids)

    print(f"\nsample : {sample!r}")
    print(f"encoded: {ids}")
    print(f"decoded: {back!r}")

    # The round-trip MUST be lossless: decode(encode(x)) == x
    assert back == sample, "round-trip failed: decode(encode(x)) != x"
    # Every ID must be a valid index into the vocab.
    assert all(0 <= i < tok.vocab_size for i in ids), "id out of range"
    print("\nOK: round-trip is lossless")
