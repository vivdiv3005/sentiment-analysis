# 🧠 Social Sentiment Analyzer

A Streamlit app that analyzes human sentiment from X/Twitter, Instagram, and YouTube posts using Claude AI.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud (free)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo and set `app.py` as the entry point
4. Add your Anthropic API key via the sidebar at runtime (no env vars needed)

## Usage
1. Enter your Anthropic API key in the sidebar
2. Paste any X, Instagram, or YouTube post URL
3. Click **Analyze ↗**

## What you get
- Overall sentiment label + confidence
- Emotion breakdown (Joy, Anger, Sadness, Trust, etc.)
- Detected themes
- Human behavioral insights
- Toxicity score & engagement prediction
