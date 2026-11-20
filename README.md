---
title: DarGlobal Wasalt Chatbot
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# DarGlobal & Wasalt RAG Chatbot

Chatbot answering questions about DarGlobal and Wasalt from publicly scraped pages.

- **Scraping:** `scraper.py` (requests + BeautifulSoup, respects robots.txt, 1 s delay) -> `data/*.json`
- **Retrieval:** BM25 over ~900-char chunks (no embedding model, small image, fast cold start)
- **LLM:** free model via OpenRouter, discovered at runtime with fallbacks if one is rate-limited
- **Serving:** FastAPI + single-page chat UI, containerised with Docker
- **Config:** env var `OPENROUTER_API_KEY` (required), `OPENROUTER_MODEL` (optional override)

Run locally: `docker build -t chatbot . && docker run -p 7860:7860 -e OPENROUTER_API_KEY=... chatbot`