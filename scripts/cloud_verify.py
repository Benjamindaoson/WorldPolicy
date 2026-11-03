"""SHA256 manifest for artifact transfer; generate locally and verify on Linux/cloud."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def entries(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        if args.manifest.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Manifest must be outside hashed root")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"schema": 1, "files": entries(args.root)}, indent=2))
        print(f"Hashed {len(entries(args.root))} files")
    else:
        expected = json.loads(args.manifest.read_text())
        if expected["schema"] != 1 or entries(args.root) != expected["files"]:
            raise SystemExit("Transfer integrity failed: missing, extra or changed file")
        print(f"Verified {len(expected['files'])} files")


if __name__ == "__main__":
    main()
