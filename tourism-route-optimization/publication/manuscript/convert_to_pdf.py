"""Convert the manuscript markdown to a journal-style PDF using markdown_pdf.

Usage: python convert_to_pdf.py
Output: MANUSCRIPT_Tourism_Route_Optimization_Yogyakarta.pdf in same directory.
"""
import os
import re
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(SCRIPT_DIR, "MANUSCRIPT_Tourism_Route_Optimization_Yogyakarta.md")
PDF_FILE = os.path.join(SCRIPT_DIR, "MANUSCRIPT_Tourism_Route_Optimization_Yogyakarta.pdf")
FIG_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "results", "figures"))

# Journal-style CSS
CSS = """
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.45;
    text-align: justify;
}
h1 {
    font-size: 15pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 6pt;
}
h2 {
    font-size: 12pt;
    font-weight: bold;
    text-align: center;
    margin-top: 16pt;
    margin-bottom: 8pt;
}
h3 {
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 12pt;
    margin-bottom: 4pt;
}
h4 {
    font-size: 11pt;
    font-style: italic;
    margin-top: 8pt;
    margin-bottom: 4pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 8pt 0;
}
th {
    font-weight: bold;
    padding: 3pt 5pt;
    border-top: 2px solid #000;
    border-bottom: 1px solid #000;
    text-align: center;
}
td {
    padding: 2pt 5pt;
    text-align: center;
}
tbody tr:last-child td {
    border-bottom: 2px solid #000;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10pt auto;
}
hr {
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 14pt 0;
}
"""


def convert():
    from markdown_pdf import MarkdownPdf, Section

    print(f"Reading {MD_FILE}...")
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    # markdown_pdf resolves image paths relative to the `root` parameter.
    # Replace all relative paths with just the filename, then set root=FIG_SRC.
    def simplify_img_path(match):
        alt = match.group(1)
        rel_path = match.group(2)
        filename = os.path.basename(rel_path)
        abs_check = os.path.join(FIG_SRC, filename)
        exists = os.path.exists(abs_check)
        print(f"  {filename} -> exists={exists}")
        return f"![{alt}]({filename})"

    md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", simplify_img_path, md_text)

    print("Creating PDF...")
    pdf = MarkdownPdf(toc_level=0)
    pdf.meta["title"] = "Tourism Route Optimization Yogyakarta"
    pdf.meta["author"] = ""

    section = Section(
        md_text,
        toc=False,
        root=FIG_SRC,  # image paths resolve from here
        paper_size="A4",
        borders=(36, 36, -36, -36),
    )
    pdf.add_section(section, user_css=CSS)
    pdf.save(PDF_FILE)

    file_size = os.path.getsize(PDF_FILE)
    print(f"\nDone! PDF size: {file_size / 1024:.0f} KB ({file_size / (1024*1024):.1f} MB)")
    print(f"Output: {PDF_FILE}")


if __name__ == "__main__":
    convert()
