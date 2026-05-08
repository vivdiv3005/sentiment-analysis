# 🧠 Social Sentiment Analyzer — Streamlit Cloud Edition

Instant sentiment analysis for X, Instagram, and YouTube posts.
**No API key. No cold starts. 100% local libraries.**

## Stack
- **VADER** (`vaderSentiment`) — rule-based, built for social media, emojis, slang
- **TextBlob** — polarity + subjectivity scoring
- **BeautifulSoup** — YouTube meta tag scraping (YouTube only)

## Deploy to Streamlit Cloud (free)

1. Push this entire folder to a GitHub repo
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch, and set **Main file path** to `app.py`
4. Click **Deploy** — live in ~60 seconds, no secrets needed

## Run locally

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one-time NLTK corpus download
streamlit run app.py
```

## How it works

| Step | What happens |
|---|---|
| User pastes URL | Platform detected (X / Instagram / YouTube) |
| User pastes text | X & Instagram text pasted manually (they block scraping) |
| YouTube | Title + description auto-fetched via og: meta tags |
| Analysis | VADER compound score + TextBlob polarity/subjectivity |
| Emotions | Derived from VADER scores + keyword boosters |
| Themes | Keyword matching across 10 theme categories |
| Insights | Rule-based linguistic pattern detection |
