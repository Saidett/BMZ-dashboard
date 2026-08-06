import requests
import json
import time
from tqdm import tqdm

# Using API of the Aurora model because there is no better interface
url = "https://aurora-sdg.labs.vu.nl/classifier/classify/aurora-sdg-multi"

# load chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

headers = {'Content-Type': 'application/json'}

for i, chunk in enumerate(tqdm(loaded_chunks)):
    
    # send API request with chunk text
    payload = json.dumps({"text": chunk["text"]})
    
    for attempt in range(5):
        try:
            response = requests.request("POST", url, headers = headers, data = payload, timeout = 300)
            if response.status_code == 429:                       # rate limit specifically
                raise requests.exceptions.HTTPError("HTTP 429 (rate limited)")
            break
        except requests.exceptions.RequestException as e:          # network/timeout/http
            wait = 2 ** attempt
            print(f"attempt {attempt+1}/{5} failed ({e}); retry in {wait}s")
            time.sleep(wait)
    else: 
        raise RuntimeError(f"failed after {5} attempts")    # crash-safe: resume later
    
    response_json = json.loads(response.text)

    # format response into a simple dict, keeping only SDG goal and prediction certainty
    goals = {p["sdg"]["code"]: p["prediction"] for p in response_json["predictions"]}

    # keep only goals with at least 0.10 certainty and sort
    top_goals = sorted({k:v for (k,v) in goals.items() if v > 0.10}.items(), key = lambda item: item[1], reverse = True)

    loaded_chunks[i]["candidates"] = top_goals

    # limitting API calls to 5 per second
    time.sleep(0.2)

# save classified chunks back
with open("data/processed/chunks_classified.json", "w", encoding="utf-8") as f:
     json.dump(loaded_chunks, f, ensure_ascii = False, indent = 2)