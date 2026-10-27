import re
from pathlib import Path

PATTERN = re.compile(r"(?i)product_[a-z]|panda|franka|\bpeg\d|\bcover\b")


def test_no_product_or_arm_names_in_src():
    src = Path(__file__).parent.parent / "src"
    hits = [
        f"{p}:{i}: {line.strip()}"
        for p in src.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if PATTERN.search(line)
    ]
    assert list(src.rglob("*.py")), "no files scanned"
    assert not hits, hits
