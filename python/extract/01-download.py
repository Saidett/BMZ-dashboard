import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os
from python.config import DATA_RAW

# publications base url
base_url = "https://www.transparenzportal.bund.de/api/v1/publications?"

# get the json with publications
read = requests.get(base_url).json()

# how many hits?
read["meta"]["count"]

# keep only documents with strategy
document_categories = ["B11", "B02", "B04", "B12", "B01", "B18"]

# keep only German-language publications
german_pubs = [p for p in read["data"] if p["language"] == "de" and p["category"] in document_categories]

urls = [dict["url"] for dict in german_pubs]

# define list to save results
publications_list = []

# cycle through base url with offset and limit at 9, then append
for offset in range(0, nr_publications, 9):
    url = f"{base_url}?limit=9&offset={offset}"
    read = requests.get(url)
    soup = BeautifulSoup(read.content, "html.parser")
    
    links = soup.find_all("a", class_="a-publication-button a-publication-button--with-clickarea")
    hrefs = [link.get("href") for link in links]
    publications_list.extend(hrefs)

# remove kinderbuch and stickerbuch and kurzfassung
terms_list = ["kinderbuch", "stickerbuch", "kinderplakat", "kurzfassung", "umwelterklaerung"]
filtered_list = [url for url in publications_list 
            if not any(term in url.lower() for term in terms_list)]

print(len(filtered_list))

for pdf_file in filtered_list:
    file_path = os.path.join(DATA_RAW, pdf_file.split('/')[-1])
    response = requests.get(pdf_file)
    
    file = open(file_path, "wb")
    file.write(response.content)
    file.close()
