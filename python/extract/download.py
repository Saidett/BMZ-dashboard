import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os
from python.config import DATA_RAW, BMZ_URL

# publications base url
base_url = BMZ_URL

# get the url from requests get method
read = requests.get(base_url)

# full html content 
html_content = read.content

# parse the html content 
soup = BeautifulSoup(html_content, "html.parser")

# find number of publications in total
nr_publications = int(soup.select_one("div[data-hits]")["data-hits"])

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
filter_terms = ["kinderbuch", "stickerbuch", "kinderplakat", "kurzfassung"]
filtered_list = [url for url in publications_list 
            if not any(term in url.lower() for term in filter_terms)]

print(len(filtered_list))

for pdf_file in filtered_list:
    file_path = os.path.join(DATA_RAW, pdf_file.split('/')[-1])
    response = requests.get(pdf_file)
    
    file = open(file_path, "wb")
    file.write(response.content)
    file.close()