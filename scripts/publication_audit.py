"""Read-only public release audit of every Git blob; prints paths, never secret values."""

import re
import subprocess

PATTERNS = [re.compile(value, re.IGNORECASE) for value in (
    rb"ghp_[A-Za-z0-9]{20,}", rb"github_pat_[A-Za-z0-9_]{20,}",
    rb"AKIA[0-9A-Z]{16}", rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    rb"(?:api[_-]?key|client_secret|password)\s*[:=]\s*['\"][^'\"\r\n]{8,}",
)]


def main() -> None:
    lines = subprocess.check_output(["git", "rev-list", "--all", "--objects"]).decode().splitlines()
    seen = set()
    hits = []
    oversized = []
    for line in lines:
        digest, _, path = line.partition(" ")
        if digest in seen or subprocess.check_output(["git", "cat-file", "-t", digest]).strip() != b"blob":
            continue
        seen.add(digest)
        data = subprocess.check_output(["git", "cat-file", "blob", digest])
        if len(data) > 1_000_000:
            oversized.append({"path": path, "bytes": len(data)})
        if any(pattern.search(data) for pattern in PATTERNS):
            hits.append(path)
    print({"historical_blobs": len(seen), "secret_pattern_paths": sorted(set(hits)),
           "blobs_over_1mb": oversized})
    if hits or oversized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
