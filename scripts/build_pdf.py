#!/usr/bin/env python3
"""Render Markdown docs to PDF with headless Chrome.

Usage: python scripts/build_pdf.py FILE.md [FILE.md ...]
Output: build/pdf/<name>.pdf (and the intermediate .html next to it).

Needs the `markdown` package (pip install -e ".[report]") and Google Chrome or
Chromium. Set CHROME=/path/to/chrome if it is not found automatically. No
Chrome? Open the .html in any browser and use Print > Save as PDF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "build" / "pdf"

CSS = """
@page { size: Letter; margin: 0.9in 0.85in; }
body { font-family: "Times New Roman", Times, serif; font-size: 11.5pt; line-height: 1.4;
       color: #111; max-width: 7in; margin: 0 auto; }
h1 { font-family: Helvetica, Arial, sans-serif; font-size: 22pt; margin: 0 0 4pt; }
h2 { font-family: Helvetica, Arial, sans-serif; font-size: 15pt; margin: 18pt 0 6pt;
     border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-family: Helvetica, Arial, sans-serif; font-size: 12pt; margin: 14pt 0 4pt; }
h4 { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; margin: 10pt 0 2pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 3pt 7pt; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; background: #f5f5f5;
       padding: 0 2pt; }
pre { background: #f5f5f5; padding: 6pt 8pt; font-size: 9pt; white-space: pre-wrap;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #ccc; margin: 6pt 0; padding: 0 10pt; color: #444; }
.page-break { page-break-after: always; }
"""

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_chrome() -> str | None:
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    return None


def to_html(md_path: Path) -> Path:
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    base = md_path.parent.resolve().as_uri() + "/"
    html = (
        f'<!doctype html><html><head><meta charset="utf-8"><base href="{base}">'
        f"<title>{md_path.stem}</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{md_path.stem}.html"
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html_path: Path, chrome: str) -> Path:
    pdf = html_path.with_suffix(".pdf")
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            f"--print-to-pdf={pdf}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    return pdf


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    chrome = find_chrome()
    for arg in argv:
        html = to_html(Path(arg))
        if chrome is None:
            print(f"{html} (Chrome not found: open it in a browser and print to PDF)")
            continue
        print(to_pdf(html, chrome).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
