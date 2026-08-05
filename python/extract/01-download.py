import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os
from python.config import DATA_RAW
import time
import pickle

# publications base url
base_url = "https://www.transparenzportal.bund.de/api/v1/publications?"

# get the json with publications
read = requests.get(base_url).json()

# keep only documents with strategy, evaluations and results
document_categories = ["B11", "B02", "B04", "B12", "B01", "A04", "A08", "A07", "B09"]

# keep only German-language publications and only pdfs
german_pubs = [p for p in read["data"] if p["language"] == "de" and p["category"] in document_categories and ".pdf" in p["url"]]
len(german_pubs)

# remove irrelevant keys
for d in german_pubs:
    d.pop("mimetype", None)

for publication in german_pubs:
    
    url = publication["url"]
    file_path = os.path.join(DATA_RAW, url.split('/')[-1])
    
    # tries 5 times, if good response, continues, if not sleep and retry
    for attempt in range(5):
        try: 
            response = requests.get(url, timeout = 120)
            break
        except requests.exceptions.ChunkedEncodingError:
            time.sleep(2 ** attempt)

    file = open(file_path, "wb")
    file.write(response.content)
    file.close()

    time.sleep(0.2)

# append file path to each dict in publication list
for pub in german_pubs:
    pub["path"] = os.path.join(DATA_RAW, pub["url"].split('/')[-1])

# save file with publication meta data
with open("data/processed/publication_meta.json", "w", encoding = "utf-8") as f:
    json.dump(german_pubs, f, ensure_ascii = False, indent = 2)