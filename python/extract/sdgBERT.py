import requests
import json
from operator import itemgetter
import time

# Using API of the Aurora model because there is no better interface
url = "https://aurora-sdg.labs.vu.nl/classifier/classify/aurora-sdg-multi"

# load chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

headers = {'Content-Type': 'application/json'}

for i, chunk in enumerate(loaded_chunks):
    
    # send API request with chunk text
    payload = json.dumps({"text": chunk["text"]})
    response = requests.request("POST", url, headers = headers, data = payload)
    response_json = json.loads(response.text)

    # format response into a simple dict, keeping only SDG goal and prediction certainty
    goals = {p["sdg"]["code"]: p["prediction"] for p in response_json["predictions"]}

    # keep only goals with at least 0.10 certainty and sort
    top_goals = sorted({k:v for (k,v) in goals.items() if v > 0.10}.items(), key = lambda item: item[1], reverse = True)

    # remove certainty value then append to chunks list
    top_goals = [g for (g, p) in top_goals]

    loaded_chunks[i]["candidates"] = top_goals

    # limitting API calls to 5 per second
    time.sleep(0.2)