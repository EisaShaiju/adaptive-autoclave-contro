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
table.eqtable { width: 100%; margin-top: 9pt; margin-bottom: 9pt; }
td.eqcell { text-align: center; border-bottom: 0px; padding: 0pt; }
td.eqnum  { text-align: right; border-bottom: 0px; padding: 0pt;
            font-size: 9.7pt; width: 34pt; }
"""

# --------------------------------------------------------------------------
# Math handling
#
#   * DISPLAY equations ($$...$$) are typeset properly by matplotlib's mathtext
#     engine and embedded as images -- Unicode approximation is not good enough
#     for \sum with limits, \frac, norms, or stacked sub/superscripts.
#   * INLINE math ($...$) is converted to HTML with real <sub>/<sup> tags,
#     which keeps the text flowing and selectable.
# --------------------------------------------------------------------------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ",
    "Delta": "Δ", "epsilon": "ε", "rho": "ρ", "sigma": "σ", "Sigma": "Σ",
    "lambda": "λ", "Lambda": "Λ", "mu": "μ", "tau": "τ", "phi": "φ",
    "Phi": "Φ", "theta": "θ", "omega": "ω", "Omega": "Ω", "pi": "π",
}
# NOTE: matched longest-first at substitution time, so that e.g. \top is not
# eaten by \to (which produced "z^→p" for z^\top before this was fixed).
SYMS = {
    # \top -> plain "T"; a preceding ^ then turns it into a real <sup>T</sup>.
    r"\top": "T", r"\times": "×", r"\cdot": "·", r"\approx": "≈",
    r"\propto": "∝", r"\infty": "∞", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\to": "→", r"\in": "∈",
    r"\sum": "Σ", r"\star": "*", r"\pm": "±", r"\ldots": "…", r"\dots": "…",
    r"\|": "‖", r"\quad": " ", r"\qquad": "  ", r"\,": " ", r"\;": " ",
    r"\!": "", r"\ ": " ",
}
_SYM_RE = re.compile("|".join(re.escape(k) for k in sorted(SYMS, key=len, reverse=True)))


def demath(s: str) -> str:
    """Inline LaTeX-ish fragment -> HTML using <sub>/<sup> and Unicode symbols.

    Used for $...$ spans only; display equations go through mathtext instead.
    """
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\mathbb\{R\}", "ℝ", s)
    s = re.sub(r"\\(?:operatorname|text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[td]?frac(\d)(\d)", r"\1/\2", s)   # brace-less \tfrac12
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    # \bar over a Greek command (\bar\alpha) must resolve the name first,
    # otherwise the generic \command rule below leaves a literal "bar".
    s = re.sub(r"\\bar\s*\\([A-Za-z]+)",
               lambda m: GREEK.get(m.group(1), m.group(1)) + "\u0304", s)
    s = re.sub(r"\\bar\s*\{?\s*([A-Za-z])\s*\}?", lambda m: m.group(1) + "\u0304", s)
    s = _SYM_RE.sub(lambda m: SYMS[m.group(0)], s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), m.group(1)), s)

    s = html.escape(s)
    # braced scripts first, then single-character ones
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9\u2074-\u209f\-+])", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9\-+])", r"<sub>\1</sub>", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def latex_for_mathtext(s: str) -> str:
    """Normalise a display equation into matplotlib's mathtext subset."""
    s = re.sub(r"\\tag\{([^}]*)\}", "", s)
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", s)
    s = s.replace(r"\quad", r"\ \ ").replace(r"\qquad", r"\ \ \ \ ")
    # mathtext knows \leq/\geq/\neq but not the short \le/\ge/\ne aliases.
    s = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", s)
    s = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", s)
    s = re.sub(r"\\ne(?![a-zA-Z])", r"\\neq", s)
    return re.sub(r"\s+", " ", s).strip()


def render_equation(latex: str, out_png: Path, fontsize=11.5, dpi=460) -> bool:
    """Typeset one display equation to a transparent PNG. False if unrenderable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = latex_for_mathtext(latex)
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0, 0, f"${body}$", fontsize=fontsize)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                    transparent=True)
        return True
    except Exception:
        return False
    finally:
        plt.close(fig)


def inline(text: str) -> str:
    """Markdown inline spans -> HTML; $...$ math -> sub/sup HTML."""
    slots = []

    def _stash(m):
        slots.append(demath(m.group(1)))
        return f"\x00{len(slots)-1}\x01"

    text = re.sub(r"\$([^$]+)\$", _stash, text)
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
    return re.sub(r"\x00(\d+)\x01", lambda m: "<i>" + slots[int(m.group(1))] + "</i>", s)


def md_to_html(md: str, img_dir: Path = None) -> str:
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    title_done = False
    eq_no = 0

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

            tag = re.search(r"\\tag\{([^}]*)\}", body)
            label = tag.group(1) if tag else None

            placed = False
            if img_dir is not None:
                eq_no += 1
                png = img_dir / f"eq{eq_no:02d}.png"
                if render_equation(body, png):
                    # Width in points: cap so wide constraint rows still fit.
                    import PIL.Image as _Image
                    with _Image.open(png) as im:
                        w_pt = im.width * 72.0 / 460.0
                    max_w = (PAGE_W - 2 * MARGIN_X) - 34
                    w_pt = min(w_pt, max_w)
                    num = (f'<td class="eqnum">({label})</td>' if label
                           else '<td class="eqnum"></td>')
                    out.append(
                        '<table class="eqtable"><tr><td class="eqcell">'
                        f'<img src="{png.name}" width="{w_pt:.1f}"/>'
                        f"</td>{num}</tr></table>")
                    placed = True
            if not placed:
                shown = demath(re.sub(r"\\tag\{[^}]*\}", "", body))
                suffix = f"   ({label})" if label else ""
                out.append('<p class="eq"><i>' + shown + html.escape(suffix) + "</i></p>")
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
    import tempfile, shutil
    img_dir = Path(tempfile.mkdtemp(prefix="eqimg_"))
    try:
        return _render(md_path, pdf_path, img_dir)
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def _render(md_path: Path, pdf_path: Path, img_dir: Path):
    md = md_path.read_text(encoding="utf-8")
    # "§3.1" reads as a stray glyph in body text; spell it out.
    md = re.sub(r"§§\s*", "Sections ", md)
    md = re.sub(r"§\s*", "Section ", md)
    html_doc = md_to_html(md, img_dir=img_dir)
    story = fitz.Story(html=html_doc, user_css=CSS,
                        archive=fitz.Archive(str(img_dir)))

    mediabox = fitz.Rect(0, 0, PAGE_W, PAGE_H)
    frame = fitz.Rect(MARGIN_X, MARGIN_TOP, PAGE_W - MARGIN_X, PAGE_H - MARGIN_BOT)

    # Story content is emitted through a DocumentWriter: each page supplies a
    # device that the story draws onto.
    # Build into a private temp file, then move into place. Keeps a viewer that
    # currently has the target open from blocking the whole build.
    import tempfile as _tf
    tmp_pdf = Path(_tf.mkdtemp(prefix="pdfbuild_")) / pdf_path.name
    writer = fitz.DocumentWriter(str(tmp_pdf))
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
    doc = fitz.open(str(tmp_pdf))
    for k, page in enumerate(doc, start=1):
        page.insert_text(fitz.Point(PAGE_W / 2 - 4, PAGE_H - 46), str(k),
                         fontname="tiro", fontsize=9.5)
    doc.subset_fonts()
    doc.save(str(tmp_pdf), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

    # Windows: a freshly-written file is often held briefly by the indexer,
    # antivirus, or OneDrive sync, so retry before giving up.
    import os, time
    for attempt in range(12):
        try:
            os.replace(tmp_pdf, pdf_path)
            break
        except PermissionError:
            time.sleep(0.5)
    else:
        raise SystemExit(
            f"Cannot write {pdf_path} -- it is locked by another program. "
            f"Close any PDF viewer showing it and re-run; "
            f"the finished build is at {tmp_pdf}")
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
