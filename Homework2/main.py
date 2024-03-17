from generate import doit
from pdflatex import PDFLaTeX as pdf

doit()
with open("artifacts/table.tex", 'rb') as f:
    pdf.from_binarystring(f.read(), "artifacts/final.pdf").create_pdf(keep_pdf_file=True)
