#!/usr/bin/env python3
"""Stamp the SCAI 2026 camera-ready PDF with a self-archiving notice.

Produces the author-accepted-version copy for the public GitHub repo.
The source camera-ready PDF is left untouched.

Usage:
    python stamp_preprint.py SRC.pdf OUT.pdf            # pre-publication notice
    python stamp_preprint.py SRC.pdf OUT.pdf 10.1007/xx # after DOI is assigned

Requires: pip install pypdf reportlab
"""
import io
import sys

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


def make_stamp(page_w: float, page_h: float, doi: str | None) -> PdfReader:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))
    grey = Color(0.35, 0.35, 0.35)
    c.setFillColor(grey)
    c.setFont("Helvetica", 7)

    line1 = ("Author-accepted version. Accepted at the Scandinavian Conference "
             "on Artificial Intelligence (SCAI 2026), Odense, Denmark. "
             "To appear in Springer CCIS.")
    if doi:
        line2 = (f"The final authenticated version is available online at "
                 f"https://doi.org/{doi}. "
                 "\u00a9 Springer Nature Switzerland AG 2026.")
    else:
        line2 = ("The final authenticated version will be available online at "
                 "link.springer.com (DOI to follow). "
                 "\u00a9 Springer Nature Switzerland AG 2026.")

    c.drawCentredString(page_w / 2, 34, line1)
    c.drawCentredString(page_w / 2, 24, line2)
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    doi = sys.argv[3] if len(sys.argv) > 3 else None

    reader = PdfReader(src)
    writer = PdfWriter()

    first = reader.pages[0]
    w = float(first.mediabox.width)
    h = float(first.mediabox.height)
    stamp = make_stamp(w, h, doi).pages[0]

    for i, page in enumerate(reader.pages):
        if i == 0:
            page.merge_page(stamp)
        writer.add_page(page)

    meta = {
        "/Title": ("Personalised Federated Learning for District Heating "
                   "Demand Estimation under Portfolio Heterogeneity "
                   "(author-accepted version, SCAI 2026)"),
        "/Author": "Varsha Kiran Badgujar; Gideon Mbiydzenyuy",
        "/Subject": "SCAI 2026, Springer CCIS (to appear)",
    }
    if doi:
        meta["/Subject"] = f"SCAI 2026, Springer CCIS, doi:{doi}"
    writer.add_metadata(meta)

    with open(out, "wb") as fh:
        writer.write(fh)
    print(f"Stamped preprint written to {out} "
          f"({len(reader.pages)} pages, page size {w:.0f}x{h:.0f} pt, "
          f"DOI={'pending' if not doi else doi})")


if __name__ == "__main__":
    main()
