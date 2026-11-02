"""Download the Braatz Tennessee Eastman dataset (44 .dat files, ~25 MB)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(22):
        for suffix in ("", "_te"):
            name = f"d{i:02d}{suffix}.dat"
            target = out / name
            if target.exists():
                continue
            urllib.request.urlretrieve(f"{BASE}/{name}", target)
            print(f"downloaded {name}")
    print(f"Data ready in {out.resolve()}")


if __name__ == "__main__":
    main()
