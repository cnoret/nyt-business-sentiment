"""Command line entry point.

Examples::

    nyt-sentiment snapshot --months 12        # backfill a year into ./data
    nyt-sentiment snapshot                    # refresh the last two months
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nyt_sentiment.config import get_settings
from nyt_sentiment.nyt_api import months_between


def recent_months(n: int, today: pd.Timestamp | None = None) -> list[tuple[int, int]]:
    """The ``n`` most recent calendar months, current month included."""
    end = (today or pd.Timestamp.now(tz="UTC")).tz_localize(None).normalize()
    start = (end.to_period("M") - (n - 1)).to_timestamp()
    return months_between(start, end)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyt-sentiment")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="Update the Parquet snapshot used by the app.")
    snap.add_argument("--months", type=int, default=2, help="Months to (re)fetch.")
    snap.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if args.data_dir:
        settings = replace(settings, data_dir=args.data_dir)

    from nyt_sentiment.snapshot import update_snapshot

    metadata = update_snapshot(settings, recent_months(args.months))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
