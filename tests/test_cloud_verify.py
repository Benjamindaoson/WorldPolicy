import json

from scripts.cloud_verify import entries


def test_transfer_hashes_detect_changes(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"original")
    first = entries(root)
    assert list(first) == ["checkpoint.pt"]
    (root / "checkpoint.pt").write_bytes(b"changed")
    assert entries(root) != first
    assert json.dumps(first)
