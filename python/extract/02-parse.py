import pdfplumber
from pathlib import Path
from python.config import DATA_RAW
import fitz
import re
import html
import json
import unicodedata
from unicodedata import category
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from nltk.corpus import stopwords

# define helper functions
# look up for unicode characters
_ZS = {i: " " for i in range(0x110000) if category(chr(i)) == "Zs"}

# pre-cleaning helper function
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)  # remove html chars     
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_ZS)         
    text = re.sub(r"[ \t]+", " ", text)     # remove double spaces
    text = re.sub(r"[ \t]+$", "", text, flags = re.M) # remove whitespace after lines 
    text = re.sub(r"\n{3,}", "\n\n", text)  # remove excessive line breaks
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # remove special chars
    text = text.replace("\xad\n", "-\n").replace("\xad", "")   # remove soft hyphens
    text = re.sub(r"(?<=\w)\n(?=[a-zäöüß])", "", text) # if there is no space before a line break and immediately a letter (broken hyphenation) then just remove it
    return text

def dehyphenate(text):
    text = re.sub(r"-\n(?=[a-zäöüß])", "", text)              # word split
    text = re.sub(r"-\n(?=[A-ZÄÖÜ0-9])", "-", text)           # compound (EU-Staaten)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# post cleaning helper function
def final_clean(chunk):
    chunk = chunk.strip().replace("\n", " ")
    chunk = re.sub(r" +", " ", chunk)
    return chunk

# define splitter function
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200,
    length_function = len,
    add_start_index = True,
    separators = ["\n\n", "\n", ". ", " ", ""]
)

def enforce_min(chunks, min_chars=500):
    out = []
    for c in chunks:
        if out and len(c) < min_chars:
            out[-1] = out[-1] + " " + c     # merge tiny into previous
        else:
            out.append(c)
    return out

pdf_dir = DATA_RAW
all_chunks = []

# loop through PDFs to turn into chunks and save them in all_chunks list
for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    
    # using fitz to read columns properly
    doc = fitz.open(pdf_path)
    text = ""
        
    for page in doc:
        
        # gets blocks per page with coordinates
        blocks = page.get_text("blocks")
        
        # sorting blocks to sort text
        blocks.sort(key = lambda b: (b[1], b[0]))
        for b in blocks:
            if b[6] == 0:  # to keep only text blocks
                text += b[4] + "\n"
    
    # clean text before chunking
    text = clean_text(text)
    text = dehyphenate(text)

    # now do the recursive chunking with langchain
    chunk_list = splitter.split_text(text)
    
    large_chunks = []

    # ensure minimum character size of chunk, if not, merge with previous
    for c in chunk_list:
        if large_chunks and len(c) < 500:
            large_chunks[-1] = large_chunks[-1] + " " + c     # merge tiny into previous
        else:
            large_chunks.append(c)

    for chunk in large_chunks:
        all_chunks.append({
            "pdf": pdf_path.name,
            "text": final_clean(chunk),
            "char_count": len(chunk),
        })

len(all_chunks)

# remove meaningless chunks by checking stop word ratio and other
# nltk.download("stopwords")
all_stops = set(stopwords.words("german") + stopwords.words("english"))

# define function to determine if text is meaningful based on stop word ratio / too few words
def is_meaningful(text):
    words = text.split()

    # remove chunks less than 20 words
    if len(words) < 20:
        return False

    # reject spaced-out letters if more than 50% of words
    single_chars = sum(1 for w in words if len(w) == 1)
    if single_chars / len(words) > 0.50:
        return False

    # reject text without enough stop words
    words_lower = [w.lower() for w in words]
    stop_count = sum(1 for w in words_lower if w in all_stops)
    if stop_count / len(words) < 0.15:
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

# length: 33241
len(good_chunks)

# add chunk index now after !! filtering
for i, chunk in enumerate(good_chunks):
    chunk["index"] = i

# now save chunks: document name, chunk index, chunk content, character count
with open("data/processed/chunks.json", "w", encoding = "utf-8") as f:
     json.dump(good_chunks, f, ensure_ascii = False, indent = 2)
