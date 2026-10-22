import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_text", ROOT / "scripts" / "check_text.py")
assert spec and spec.loader
check_text = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_text)


def problems_for(tmp_path: Path, text: str) -> list[str]:
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    return check_text.check_file(path)


def test_flags_em_dash(tmp_path):
    found = problems_for(tmp_path, "fine line\nthe model \u2014 it works\n")
    assert len(found) == 1
    assert ":2:" in found[0] and "em dash" in found[0]


def test_flags_spaced_en_dash_but_allows_ranges(tmp_path):
    assert problems_for(tmp_path, "Data mining \u2013 clustering\n")
    assert not problems_for(tmp_path, "pages 15447\u201315451\n")


def test_flags_invisible_characters(tmp_path):
    assert problems_for(tmp_path, "zero\u200bwidth\n")
    assert problems_for(tmp_path, "non\u00a0breaking\n")


def test_ignore_marker_skips_the_line(tmp_path):
    assert not problems_for(tmp_path, "quoted \u2014 dash  <!-- text-check: ignore -->\n")


def test_binary_files_are_skipped(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\xff\xfe\x00\x81")
    assert check_text.check_file(path) == []
