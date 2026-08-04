# Dashboard to summarise and visualise BMZ projects and data OUTDATED README

An interactive dashboard exploring Germany's development cooperation strategy and projects, combining BMZ policy documents with IATI data.

## Overview

This project scrapes, parses, and classifies BMZ publications to make German development cooperation strategy searchable and explorable. The dashboard allows users to:

- Browse BMZ strategy documents by **SDG topic area**
- View relevant policy text chunks for each topic
- Explore **IATI aid data** filtered by country and sector

## Architecture

BMZ PDFs (data/raw/) → Parse & chunk (python/extract/parse.py) → Classify by SDG using Aurora SDG multilabel model → Classify by country and region (upcoming) → Merge & aggregate (python/extract/merge.py) → JSON/CSV outputs (data/processed/)

IATI data → Country-level project data, funding, sectors

R Shiny Dashboard (R/) → Reads pre-processed JSON → Interactive filtering by SDG, country, document

## Data Sources

- **BMZ Publications**: strategy papers, position papers, and reports scraped from [bmz.de](https://www.bmz.de)
- **IATI**: Aid activity data from the [IATI Registry](https://www.iatiregistry.org/)

## Pipeline

1. **Download** — Scrape BMZ publication PDFs (`download.py`)
2. **Parse** — Extract text and split into chunks (`parse.py`)
3. **Classify** — Label each chunk with SDG numbers using LLM (`classify.py`)
4. **Merge** (in progress) — Aggregate chunk classifications to document level (`merge.py`)

## Tech Stack

- **Python**: PDF parsing (PyMuPDF)
- **R**: Shiny dashboard
- **Models** (via Ollama): llama3.2:3b (classification), nomic-embed-text (embeddings) OUTDATED

## Project Structure
```
├── python/
│   ├── config.py              # Paths and constants
│   └── extract/
│       ├── download.py        # PDF scraper
│       ├── parse.py           # Text extraction & chunking
│       ├── classify.py        # SDG classification via LLM
│       └── merge.py           # Chunk → document aggregation
├── R/
│   └── app.R                  # Shiny dashboard
├── data/
│   ├── raw/                   # Scraped PDFs
│   └── processed/             # JSON outputs
└── pyproject.toml
```
## Getting Started

```bash
git clone <repo>
cd BMZ-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -e .
ollama pull llama3.2:3b
ollama pull nomic-embed-text
