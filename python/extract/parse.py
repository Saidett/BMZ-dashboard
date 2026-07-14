import pdfplumber
from pathlib import Path
from python.config import DATA_RAW
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
import html
import json
import nltk
from nltk.corpus import stopwords

pdf_dir = DATA_RAW
all_chunks = []

# using langchain recursive splitting to respect PDF structure
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   
    chunk_overlap=200
)

# pre-cleaning helper function
def clean_text(text):
    text = html.unescape(text)  # remove html chars            
    text = re.sub(r"[ \t]+", " ", text)     # remove double spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # remove excessive line breaks
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # remove special chars
    text = text.replace("\xad", "")           # remove soft hyphens
    text = re.sub(r"-\n", "", text)           # join hyphenated line breaks
    text = re.sub(r"(?<=\w)\n(?=\w)", "", text)  # merge line break if next to them its not whitespace (to merge hyphenated words)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single \n → space, keep \n\n
    return text.strip(" \t")

# post cleaning helper function
def final_clean(chunk):
    chunk = chunk.strip().replace("\n", " ")
    chunk = re.sub(r" +", " ", chunk)
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

len(all_chunks)

# remove meaningless chunks by checking stop word ration
# nltk.download("stopwords")
all_stops = set(stopwords.words("german") + stopwords.words("english"))

# define function to determine if text is meaningful based on stop word ratio / too few words
def is_meaningful(text):
    words = text.split()
    if len(words) < 20:
        return False

    # reject spaced-out letters
    single_chars = sum(1 for w in words if len(w) == 1)
    if single_chars / len(words) > 0.50:
        return False

    # reject text without enough stop words
    words_lower = [w.lower() for w in words]
    stop_count = sum(1 for w in words_lower if w in all_stops)
    if stop_count / len(words) < 0.30:
        return False

    # reject flat lists: many parenthetical acronyms but no sentence structure
    paren_acronyms = len(re.findall(r"\([A-Za-z]{2,5}\)", text))
    sentences = text.count(".")
    if paren_acronyms >= 5 and sentences < 3:
        return False

    return True

good_chunks = []

for chunk in all_chunks:
    if is_meaningful(chunk["text"]):
        good_chunks.append(chunk)

len(good_chunks)

# filter out chunks that are too small to be meaningful (less than 500 characters, intuitive judgement after inspection)
good_chunks = [c for c in good_chunks if c["char_count"] >= 499]

# final length: 5205 chunks
len(good_chunks)

# now save chunks: document name, chunk index, chunk content, character count
with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
     json.dump(good_chunks, f, ensure_ascii=False, indent=2)
