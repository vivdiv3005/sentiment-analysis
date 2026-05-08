# 🧠 Social Sentiment Analyzer — Streamlit Cloud Edition

Zero-dependency sentiment analysis for X, Instagram, and YouTube posts.
No API key required. Powered by Hugging Face free Inference API.

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → New app → select your repo
3. Set **Main file path** to `app.py`
4. Click **Deploy** — done!

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

- **No torch / transformers installed** — uses HF Inference API (HTTP calls)
- Sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Emotion model:   `j-hartmann/emotion-english-distilroberta-base`
- YouTube: title + description auto-fetched via og: meta tags
- X / Instagram: user pastes the post text (these platforms block scraping)

## Optional: Avoid rate limits

Get a free token at https://huggingface.co/settings/tokens
and paste it into the sidebar. Raises limit from ~30 to ~1000 req/hr.
