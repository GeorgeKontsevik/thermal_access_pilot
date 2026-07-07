from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/kaliningrad.toml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()

