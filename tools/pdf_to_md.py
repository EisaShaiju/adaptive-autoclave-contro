#!/usr/bin/env python3
"""
pdf_to_md.py -- Convert a PDF reference paper to Markdown.

Strategy (per page):
  1. Try native text extraction (PyMuPDF). Fast and exact for normal PDFs.
  2. If a page's text is missing or garbled -- e.g. a scanned page, or an
     embedded font with no Unicode map (the `ref_docs/dufour mpc.pdf` case,
     where every character extracts as meaningless CID codes) -- fall back to
     OCR: render the page to an image and read it with RapidOCR (ONNX).

Everything is pip-installable and CPU-only; no system binaries (no Tesseract,
no Poppler) and no PyTorch are required. See tools/requirements-pdf.txt.

Caveat: OCR reconstructs body text and tables well, but only *approximates*
mathematical notation (Greek letters, sub/superscripts). For exact equations,
verify against the source or use a math-specific OCR.

Usage:
  python tools/pdf_to_md.py INPUT.pdf [-o OUTPUT.md] [--dpi 150]
         [--pages 1-5,26] [--force-ocr] [--ocr-threshold 0.60]

Examples:
  python tools/pdf_to_md.py "ref_docs/dufour mpc.pdf"
  python tools/pdf_to_md.py paper.pdf -o notes/paper.md --pages 1-12
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# ---- printable-character heuristic ------------------------------------------

_GOOD = set(".,;:()[]{}%-+=/*<>|'\"$&@#°±×·—–…")


def printable_ratio(s: str) -> float:
    """Fraction of characters that look like ordinary Latin text/punctuation.

    Native extraction of a broken-font PDF returns control codes and private-use
    glyphs, driving this ratio far below a clean page's ~0.9. We treat a low
    ratio as the signal to OCR the page instead.
    """
    if not s:
        return 0.0
    ok = sum(1 for c in s if (c.isascii() and (c.isalnum() or c.isspace())) or c in _GOOD)
    return ok / len(s)


# ---- page range parsing -----------------------------------------------------

def parse_pages(spec: str, n_pages: int) -> list[int]:
    """Parse '1-5,26,30-32' (1-indexed, inclusive) into sorted 0-indexed ints."""
    if not spec:
        return list(range(n_pages))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for p in range(int(a), int(b) + 1):
                out.add(p - 1)
        else:
            out.add(int(part) - 1)
    return sorted(p for p in out if 0 <= p < n_pages)


# ---- lazy OCR engine --------------------------------------------------------

class _OCR:
    """Lazily-constructed RapidOCR wrapper (model load is ~1s, so defer it)."""

    def __init__(self) -> None:
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:  # pragma: no cover - guidance path
                raise SystemExit(
                    "OCR is required for this PDF but RapidOCR is not installed.\n"
                    "Install the (pip-only, no system binaries) OCR stack:\n"
                    "    pip install -r tools/requirements-pdf.txt"
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    def read(self, image) -> str:
        engine = self._ensure()
        result, _ = engine(image)
        if not result:
            return ""
        # result rows are [box, text, confidence] in reading order (top->bottom)
        return "\n".join(row[1] for row in result)


def _render_page(page, dpi: int):
    """Render a PyMuPDF page to an RGB numpy array for OCR."""
    import numpy as np
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    return np.array(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))


# ---- main -------------------------------------------------------------------

def convert(pdf_path: Path, out_path: Path, dpi: int, pages_spec: str,
            force_ocr: bool, ocr_threshold: float, verbose: bool = True) -> Path:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is not installed. Install the PDF toolchain:\n"
            "    pip install -r tools/requirements-pdf.txt"
        ) from exc

    doc = fitz.open(pdf_path)
    pages = parse_pages(pages_spec, doc.page_count)
    ocr = _OCR()

    chunks: list[str] = [f"# {pdf_path.stem}\n",
                         f"*Converted from `{pdf_path.name}` ({len(pages)} of "
                         f"{doc.page_count} pages) by tools/pdf_to_md.py.*\n"]
    n_native = n_ocr = 0
    t0 = time.time()

    for idx in pages:
        page = doc[idx]
        native = page.get_text().strip()
        ratio = printable_ratio(native)
        use_ocr = force_ocr or len(native) < 20 or ratio < ocr_threshold

        if use_ocr:
            text = ocr.read(_render_page(page, dpi)).strip()
            source = "OCR"
            n_ocr += 1
        else:
            text = native
            source = "native"
            n_native += 1

        if verbose:
            print(f"  page {idx + 1:>3}: {source:<6} "
                  f"(printable={ratio:.2f}, {len(text)} chars)", file=sys.stderr)

        chunks.append(f"\n---\n\n## Page {idx + 1}  _({source})_\n\n{text}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks), encoding="utf-8")

    if verbose:
        dt = time.time() - t0
        print(f"\nDone in {dt:.1f}s: {n_native} native + {n_ocr} OCR pages "
              f"-> {out_path}", file=sys.stderr)
        if n_ocr:
            print("Note: OCR pages approximate math notation; verify equations "
                  "against the source.", file=sys.stderr)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert a PDF paper to Markdown "
                                             "(native text + OCR fallback).")
    ap.add_argument("pdf", type=Path, help="input PDF path")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output .md path (default: alongside the PDF)")
    ap.add_argument("--dpi", type=int, default=150,
                    help="render DPI for OCR pages (default 150; 200+ is slower, "
                         "marginally more accurate)")
    ap.add_argument("--pages", default="",
                    help="page subset, 1-indexed, e.g. '1-5,26' (default: all)")
    ap.add_argument("--force-ocr", action="store_true",
                    help="OCR every page even if native text looks fine")
    ap.add_argument("--ocr-threshold", type=float, default=0.60,
                    help="printable-ratio below which a page is OCR'd (default 0.60)")
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        raise SystemExit(f"No such file: {args.pdf}")
    out = args.output or args.pdf.with_suffix(".md")
    convert(args.pdf, out, args.dpi, args.pages, args.force_ocr, args.ocr_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
