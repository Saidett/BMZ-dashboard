from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" 
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OLLAMA_MODEL = "llama3.2:3b"
BMZ_URL = "https://www.bmz.de/ajax/filterlist/de/24710-24710"
CLASSIFY_PROMPT = """You are analyzing a subsection of a German development ministry publication.

Extract:
- regions
- countries
- SDGs
- reasoning

Allowed regions: Africa, Asia, Latin America, Europe, Global

SDGs:
1=No Poverty, 2=Zero Hunger, 3=Good Health and Well-being, 4=Quality Education, 5=Gender Equality
6=Clean Water and Sanitation, 7=Affordable and Clean Energy, 8=Decent Work and Economic Growth, 9=Industry, Innovation and Infrastructure, 10=Reduced Inequalities
11=Sustainable Cities and Communities, 12=Responsible Consumption and Production, 13=Climate Action, 14=Life Below Water
15=Life on Land, 16=Peace, Justice and Strong Institutions, 17=Partnerships for the goals

Rules:
- Use only the allowed region names.
- Use standard English country names (e.g., Kenya, India, Germany).
- Do not include regions or continents as countries.
- Return SDGs as numbers only.
- Include an SDG if the text clearly relates to it, even if the SDG number or name is not mentioned.
- Use [] if no region, SDG or country apply.
- "reasoning": a short 1-2 sentence explanation of your choices (keep it brief).

Examples:
{{"sdgs": [13], "regions": ["Africa"], "countries": ["Kenya"], "reasoning": "Mentions renewable energy projects in Kenya, relating to SDG 13 (Climate Action)."}}
{{"sdgs": [1, 2, 10], "regions": ["Asia", "Africa"], "countries": [], "reasoning": "Discusses poverty reduction and hunger programs across Asia and Africa, relating to SDGs 1, 2, and 10."}}
{{"sdgs": [], "regions": ["Global"], "countries": [], "reasoning": "General statement about international cooperation with no specific region or SDG."}}

Text:
{text}"""