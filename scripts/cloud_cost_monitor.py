"""Append measured elapsed GPU billing estimates during a single cloud run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.hourly_usd <= 0 or args.max_usd <= 0 or args.interval <= 0:
        raise ValueError("Invalid cloud billing inputs")
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        elapsed = time.monotonic() - start
        record = {"elapsed_gpu_hours_estimate": elapsed / 3600,
                  "accrued_gpu_usd_estimate": elapsed * args.hourly_usd / 3600,
                  "remaining_authorized_usd_estimate": max(0, args.max_usd -
                                                            elapsed * args.hourly_usd / 3600),
                  "hourly_usd": args.hourly_usd}
        with args.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
