"""
md_to_paper_pdf.py
Render a Markdown technical report to a single-column, NeurIPS-style PDF.

Self-contained: uses only PyMuPDF (already a transitive dependency of the PDF
tooling) -- no LaTeX, pandoc, or wkhtmltopdf required. Implements a small
Markdown subset sufficient for the reports in docs/: headings, paragraphs,
bold/italic/code spans, bullet and numbered lists, tables, block quotes,
horizontal rules, fenced code blocks, and inline/display math written in
LaTeX-ish notation (converted to Unicode).

Usage:
    .venv/Scripts/python.exe tools/md_to_paper_pdf.py docs/REPORT.md
    .venv/Scripts/python.exe tools/md_to_paper_pdf.py docs/REPORT.md -o out.pdf
"""
from pathlib import Path
import argparse
import html
import re
import sys

import fitz

# --------------------------------------------------------------------------
# Page geometry: US Letter with NeurIPS-like margins (single column)
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = fitz.paper_size("letter")
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 90, 78, 78

CSS = """
* { font-family: Times, serif; }
body { font-size: 9.7pt; line-height: 1.34; color: #000; text-align: justify; }
h1.title { font-size: 17pt; font-weight: bold; text-align: center;
           margin-top: 0; margin-bottom: 12pt; line-height: 1.2; }
p.author { font-size: 11pt; text-align: center; margin-top: 0; margin-bottom: 3pt; }
p.rule   { text-align: center; margin-top: 2pt; margin-bottom: 14pt; }
h2.abshead { font-size: 11pt; font-weight: bold; text-align: center;
             margin-top: 6pt; margin-bottom: 5pt; }
div.abstract { margin-left: 26pt; margin-right: 26pt; font-size: 9.2pt;
               text-align: justify; margin-bottom: 14pt; }
h2 { font-size: 12pt; font-weight: bold; margin-top: 15pt; margin-bottom: 6pt;
     text-align: left; }
h3 { font-size: 10.5pt; font-weight: bold; margin-top: 12pt; margin-bottom: 5pt;
     text-align: left; }
h4 { font-size: 9.8pt; font-weight: bold; margin-top: 10pt; margin-bottom: 4pt;
     text-align: left; }
p  { margin-top: 0; margin-bottom: 6.5pt; }
p.eq { text-align: center; margin-top: 8pt; margin-bottom: 8pt; font-size: 10pt; }
ul, ol { margin-top: 2pt; margin-bottom: 7pt; }
li { margin-bottom: 3pt; text-align: justify; }
code { font-family: Courier, monospace; font-size: 8.6pt; }
pre  { font-family: Courier, monospace; font-size: 8.2pt; background-color: #f4f4f4;
       margin-top: 6pt; margin-bottom: 7pt; text-align: left; }
blockquote { margin-left: 16pt; margin-right: 16pt; font-size: 9.2pt; }
table { width: 100%; margin-top: 6pt; margin-bottom: 8pt; font-size: 8.6pt; }
th { font-weight: bold; text-align: left; border-bottom: 1px solid #000;
     padding: 3pt; }
td { text-align: left; padding: 3pt; border-bottom: 0.4px solid #bbb; }
p.caption { font-size: 8.8pt; margin-top: 4pt; margin-bottom: 7pt; }
hr { margin-top: 9pt; margin-bottom: 9pt; }
"""

# --------------------------------------------------------------------------
# LaTeX-ish math -> Unicode
# --------------------------------------------------------------------------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ",
    "Delta": "Δ", "epsilon": "ε", "rho": "ρ", "sigma": "σ", "Sigma": "Σ",
    "lambda": "λ", "Lambda": "Λ", "mu": "μ", "tau": "τ", "phi": "φ",
    "Phi": "Φ", "theta": "θ", "omega": "ω", "Omega": "Ω", "pi": "π",
}
SYMS = {
    r"\\le": "≤", r"\\leq": "≤", r"\\ge": "≥", r"\\geq": "≥", r"\\neq": "≠",
    r"\\times": "×", r"\\cdot": "·", r"\\approx": "≈", r"\\propto": "∝",
    r"\\to": "→", r"\\infty": "∞", r"\\sum": "Σ", r"\\in": "∈",
    r"\\mathbb\{R\}": "ℝ", r"\\top": "ᵀ", r"\\star": "*", r"\\pm": "±",
    r"\\ldots": "…", r"\\quad": "  ", r"\\,": " ", r"\\;": " ", r"\\!": "",
}
SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
       "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "+": "⁺", "n": "ⁿ", "i": "ⁱ",
       "T": "ᵀ", "k": "ᵏ"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
       "7": "₇", "8": "₈", "9": "₉", "a": "ₐ", "e": "ₑ", "i": "ᵢ", "k": "ₖ",
       "m": "ₘ", "n": "ₙ", "p": "ₚ", "t": "ₜ", "c": "c", "r": "ᵣ", "x": "ₓ"}


def _script(body, table):
    return "".join(table.get(ch, ch) for ch in body)


def demath(s: str) -> str:
    """Convert a LaTeX-ish math fragment to readable Unicode text."""
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    for k, v in SYMS.items():
        s = re.sub(k, v, s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), SUP), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), SUB), s)
    s = re.sub(r"\^(\w)", lambda m: _script(m.group(1), SUP), s)
    s = re.sub(r"_(\w)", lambda m: _script(m.group(1), SUB), s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def inline(text: str) -> str:
    """Markdown inline spans -> HTML, math -> Unicode."""
    text = re.sub(r"\$([^$]+)\$", lambda m: "\x00" + demath(m.group(1)) + "\x01", text)
    parts = re.split(r"(`[^`]*`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append("<code>" + html.escape(p[1:-1]) + "</code>")
        else:
            p = html.escape(p)
            p = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", p)
            p = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", p)
            p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", p)
            out.append(p)
    s = "".join(out)
    return s.replace("\x00", "<i>").replace("\x01", "</i>")


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    title_done = False

    while i < n:
        ln = lines[i]
        st = ln.strip()

        if not st:
            i += 1
            continue

        # fenced code
        if st.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre>" + "<br/>".join(buf) + "</pre>")
            continue

        # display math -- may be a single line ($$ ... $$) or span several lines
        if st.startswith("$$"):
            if st.endswith("$$") and len(st) > 4:          # complete on one line
                body = st[2:-2]
                i += 1
            else:                                           # spans multiple lines
                body_parts = [st[2:]]
                i += 1
                while i < n:
                    cur = lines[i].strip()
                    if cur.endswith("$$"):
                        body_parts.append(cur[:-2])
                        i += 1
                        break
                    body_parts.append(cur)
                    i += 1
                body = " ".join(body_parts)
            body = re.sub(r"\\tag\{([^}]*)\}", r"   (\1)", body)
            out.append('<p class="eq"><i>' + html.escape(demath(body)) + "</i></p>")
            continue

        if st.startswith("---") and set(st) <= {"-"}:
            out.append("<hr/>")
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)", st)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1 and not title_done:
                out.append('<h1 class="title">' + inline(txt) + "</h1>")
                title_done = True
            elif txt.strip().lower() == "abstract":
                out.append('<h2 class="abshead">Abstract</h2>')
                i += 1
                buf = []
                while i < n and not re.match(r"^(#{1,4})\s+|^---", lines[i].strip()):
                    if lines[i].strip():
                        buf.append(lines[i].strip())
                    elif buf:
                        break
                    i += 1
                out.append('<div class="abstract">' + inline(" ".join(buf)) + "</div>")
                continue
            else:
                out.append(f"<h{lvl if lvl>1 else 2}>" + inline(txt) + f"</h{lvl if lvl>1 else 2}>")
            i += 1
            continue

        # author line (bold-only paragraph right after title)
        if title_done and re.fullmatch(r"\*\*[^*]+\*\*", st) and not out[-1].startswith("<h2"):
            if all("<h2" not in o for o in out):
                out.append('<p class="author">' + inline(st.strip("*")) + "</p>")
                out.append('<p class="rule">\u2014\u2014\u2014</p>')
                i += 1
                continue

        # table
        if st.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i+1].strip()):
            hdr = [c.strip() for c in st.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ["<table><tr>"] + [f"<th>{inline(c)}</th>" for c in hdr] + ["</tr>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</table>")
            out.append("".join(t))
            continue

        # blockquote
        if st.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue

        # lists
        if re.match(r"^[-*]\s+", st) or re.match(r"^\d+\.\s+", st):
            ordered = bool(re.match(r"^\d+\.\s+", st))
            tag = "ol" if ordered else "ul"
            items = []
            while i < n:
                s2 = lines[i].strip()
                m2 = re.match(r"^[-*]\s+(.*)", s2) or re.match(r"^\d+\.\s+(.*)", s2)
                if not m2:
                    if s2 and lines[i].startswith(("  ", "\t")) and items:
                        items[-1] += " " + s2
                        i += 1
                        continue
                    break
                items.append(m2.group(1))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        # paragraph (caption style for Table/Figure leads)
        buf = []
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,4})\s+|^[-*]\s+|^\d+\.\s+|^\||^>|^```|^\$\$|^---$", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        if not buf:
            # Nothing consumed: this line matched a block-start pattern that no
            # branch above claimed (e.g. a stray '|' line). Emit it as a plain
            # paragraph and advance, so the parser can never stall.
            out.append("<p>" + inline(st) + "</p>")
            i += 1
            continue
        para = " ".join(buf)
        cls = ' class="caption"' if re.match(r"^\*\*(Table|Figure)", para) else ""
        out.append(f"<p{cls}>" + inline(para) + "</p>")

    return f"<html><head><style>{CSS}</style></head><body>" + "".join(out) + "</body></html>"


def render(md_path: Path, pdf_path: Path):
    html_doc = md_to_html(md_path.read_text(encoding="utf-8"))
    story = fitz.Story(html=html_doc, user_css=CSS)

    mediabox = fitz.Rect(0, 0, PAGE_W, PAGE_H)
    frame = fitz.Rect(MARGIN_X, MARGIN_TOP, PAGE_W - MARGIN_X, PAGE_H - MARGIN_BOT)

    # Story content is emitted through a DocumentWriter: each page supplies a
    # device that the story draws onto.
    writer = fitz.DocumentWriter(str(pdf_path))
    more, pages = 1, 0
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(frame)
        story.draw(dev)
        writer.end_page()
        pages += 1
        if pages > 300:
            break
    writer.close()

    # Second pass: page numbers, centred at the foot.
    doc = fitz.open(str(pdf_path))
    for k, page in enumerate(doc, start=1):
        page.insert_text(fitz.Point(PAGE_W / 2 - 4, PAGE_H - 46), str(k),
                         fontname="tiro", fontsize=9.5)
    doc.subset_fonts()
    doc.save(str(pdf_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args()
    out = a.output or a.markdown.with_suffix(".pdf")
    pages = render(a.markdown, out)
    print(f"wrote {out}  ({pages} pages)")


if __name__ == "__main__":
    sys.exit(main())
