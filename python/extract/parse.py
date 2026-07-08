import pdfplumber
from pathlib import Path
from python.config import DATA_RAW

pdf_dir = DATA_RAW
all_chunks = []

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    
    chunks = chunk_text(text)
    
    for chunk in chunks:
        all_chunks.append({
            "pdf": pdf_path.name,
            "chunk": chunk,
            "char_count": len(chunk),
        })


