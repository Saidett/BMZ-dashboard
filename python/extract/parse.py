import pdfplumber
from pathlib import Path
from python.config import DATA_RAW
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
import html
import json

pdf_dir = DATA_RAW
all_chunks = []

# using langchain recursive splitting to respect PDF structure
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   
    chunk_overlap=50
)

# pre-cleaning helper function
def clean_text(text):
    text = html.unescape(text)  # remove html chars            
    text = re.sub(r'[ \t]+', ' ', text)     # remove double spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # remove excessive line breaks
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # remove special chars
    text = text.replace('\xad', '')           # remove soft hyphens
    text = re.sub(r'-\n', '', text)           # join hyphenated line breaks
    text = re.sub(r'(?<=\w)\n(?=\w)', '', text)  # merge line break if next to them its not whitespace (to merge hyphenated words)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # single \n → space, keep \n\n
    return text.strip(" \t")

# post cleaning helper function
def final_clean(chunk):
    chunk = chunk.strip().replace('\n', ' ')
    chunk = re.sub(r' +', ' ', chunk)
    return chunk

# loop through PDFs to turn into chunks and save them in all_chunks list
for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    
    # using fitz to read columns properly
    doc = fitz.open(pdf_path)
    text = ""
        
    for page in doc:
        # gets blocks per page with coordinates
        blocks = page.get_text("blocks")
        # sorting blocks to sort text
        blocks.sort(key=lambda b: (b[1], b[0]))
        for b in blocks:
            if b[6] == 0:  # to keep only text blocks
                text += b[4] + "\n"
    
    # clean text before chunking
    text = clean_text(text)

    # now do the recursive chunking with langchain
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "pdf": pdf_path.name,
            "chunk_index": i,
            "text": final_clean(chunk),
            "char_count": len(chunk),
        })

# now save chunks: document name, chunk index, chunk content, character count
with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
