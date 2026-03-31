# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a text mining / NLP project for **subjective analysis of Czech news articles** (Subjektivní analýza novinových článků). The goal is to detect bias and subjectivity in articles scraped from Czech news sites (e.g. seznamzpravy.cz).

Everything lives in a single Jupyter notebook: `TextMiningProjekt.ipynb`.

## Environment

- Python 3.13 via Homebrew (`/opt/homebrew/bin/python3.13`)
- Packages are installed user-wide (not in a venv) using `--user --break-system-packages`
- To install packages: `/opt/homebrew/bin/python3.13 -m pip install <pkg> --user --break-system-packages`
- To run the notebook: open in VS Code and select the Python 3.13 kernel

## Pipeline Architecture

The notebook follows this sequential pipeline:

1. **Web scraping** (cells 5–7) — `newspaper3k` + `BeautifulSoup` scrape articles from a Czech news URL into a `articles_df` DataFrame with columns: `title`, `text`, `authors`, `publish_date`, `url`
2. **Text preprocessing** (cells 9–11) — tokenization, lowercasing, punctuation removal, Czech stop word filtering → `cleaned_text` column
3. **POS tagging** (cells 13–15) — `stanza` Czech model tags each token with its part of speech → `pos_tags` column
4. **TF-IDF analysis** (cells 18–20) — `sklearn` TfidfVectorizer extracts keyword topics per article
5. **Lemmatization + bigrams** (cells 22–23) — UDPipe (`ufal.udpipe`) lemmatizes tokens; TF-IDF is rerun with bigrams → `lemmatized_text` column
6. **Word2Vec embeddings** (cells 25–26) — `gensim` Word2Vec trained on lemmatized corpus for semantic similarity
7. **Transformer sentiment** (cell 28) — HuggingFace `transformers` Czech sentiment model (requires `HF_TOKEN` secret, originally loaded via `google.colab.userdata`)
8. **Subjectivity index** (cells 30–36) — custom score based on ratio of adjectives/adverbs (ADJ/ADV POS tags) to factual tokens; refined using Word2Vec cosine distances to anchor words
9. **Comparison & visualization** (cells 38–41) — comparison of scoring approaches, normalization analysis

## Key Dependencies

Installed inline in the notebook via `!pip install`:
- `newspaper3k`, `lxml_html_clean` — web scraping
- `stanza` — Czech POS tagging (downloads `cs` model on first run)
- `ufal.udpipe` — Czech lemmatization
- `textblob`, `transformers`, `torch` — sentiment analysis
- `gensim` — Word2Vec
- `sklearn` — TF-IDF
- `pandas`, `numpy`

## Important Notes

- Most cells are guarded with `if 'articles_df' in locals() and not articles_df.empty:` — run cells in order from top to bottom.
- The transformer sentiment cell (28) uses `google.colab.userdata` to load a HuggingFace token — this will fail outside Colab. Replace with `os.environ.get('HF_TOKEN')` or hardcode for local use.
- The Word2Vec section (cell 24) is marked **PŘESKOČIT** (skip) — it depends on having `lemmatized_text` populated from UDPipe first.
- `stanza.download('cs')` only needs to run once; it caches the model locally.
