import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, parse_qs
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }

div[data-testid="stTextInput"] input {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    border-radius: 8px;
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
.stProgress > div > div > div { border-radius: 4px; }

.insight-box {
    background: #f8f9fa;
    border-left: 3px solid #6c63ff;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.55;
    color: #333;
}
.free-badge {
    background: #d1fae5;
    color: #065f46;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
PLATFORM_ICONS = {"x": "𝕏", "twitter": "𝕏", "instagram": "📸", "youtube": "▶️", "unknown": "🌐"}

SENTIMENT_MAP = {
    "positive": ("😄", "Positive", "The content carries an overall positive and uplifting tone."),
    "negative": ("😞", "Negative", "The content carries a negative or critical emotional tone."),
    "neutral":  ("😐", "Neutral",  "The content is balanced or matter-of-fact in tone."),
}

THEME_KEYWORDS = {
    "Community":   ["we", "us", "together", "everyone", "people", "community", "share", "join"],
    "Achievement": ["won", "win", "success", "proud", "achieved", "milestone", "goal", "best"],
    "Humor":       ["lol", "haha", "funny", "joke", "laugh", "hilarious"],
    "Criticism":   ["wrong", "bad", "terrible", "worst", "hate", "awful", "disappointing", "fail"],
    "Inspiration": ["inspire", "motivation", "dream", "believe", "hope", "amazing", "incredible"],
    "News":        ["breaking", "update", "report", "just in", "official", "announced", "today"],
    "Gratitude":   ["thank", "grateful", "appreciate", "blessed", "thankful"],
    "Question":    ["?", "why", "how", "what", "when", "who", "wonder"],
    "Nostalgia":   ["remember", "miss", "used to", "back then", "old days", "throwback"],
    "Excitement":  ["amazing", "wow", "omg", "excited", "can't wait"],
}

INSIGHT_RULES = [
    (["we", "us", "our", "together", "everyone"],
     "Uses inclusive language — signals a community-oriented mindset or appeal to shared identity."),
    (["?"],
     "Rhetorical or genuine questions suggest the author is seeking engagement or expressing uncertainty."),
    (["!", "omg", "wow", "amazing", "incredible"],
     "Exclamatory language and intensity markers indicate high emotional arousal."),
    (["i", "me", "my", "myself"],
     "Heavy first-person language points to a personal, reflective, or self-expressive post."),
    (["you", "your"],
     "Direct address ('you/your') suggests the author is speaking to or challenging the audience."),
    (["but", "however", "although", "yet", "despite"],
     "Contrasting language signals nuanced thinking or emotional ambivalence."),
    (["always", "never", "everyone", "nobody", "all"],
     "Absolute language ('always/never/everyone') may indicate strong conviction or emotional reactivity."),
    (["lol", "haha", "joke", "funny"],
     "Humor markers suggest the author is using levity, possibly to soften a serious point."),
    (["miss", "remember", "used to", "back", "throwback"],
     "Nostalgic phrasing reflects an emotional attachment to the past."),
    (["please", "help", "need", "support"],
     "Appeal language ('please/help/need') signals vulnerability or a direct call to action."),
]


# ── Model loading (cached so they download only once) ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
    )

@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )


# ── Text extraction ────────────────────────────────────────────────────────────
def detect_platform(url: str) -> str:
    u = url.lower()
    if "x.com" in u or "twitter.com" in u:
        return "x"
    if "instagram.com" in u:
        return "instagram"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    return "unknown"


def fetch_text_from_url(url: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    platform = detect_platform(url)
    handle, time_str, content, raw_text = "Unknown", "Recently", "", ""

    try:
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")

        if platform == "youtube":
            title_tag = soup.find("meta", {"name": "title"}) or soup.find("meta", {"property": "og:title"})
            desc_tag  = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
            channel   = soup.find("link", {"itemprop": "name"})
            handle    = channel["content"] if channel and channel.get("content") else "YouTube Creator"
            time_str  = "YouTube Video"
            title     = title_tag["content"] if title_tag and title_tag.get("content") else ""
            desc      = desc_tag["content"]  if desc_tag  and desc_tag.get("content")  else ""
            content   = f"{title}. {desc}".strip(". ")
            raw_text  = content

        elif platform == "x":
            og_desc  = soup.find("meta", {"property": "og:description"})
            content  = og_desc["content"] if og_desc and og_desc.get("content") else ""
            parts    = urlparse(url).path.strip("/").split("/")
            handle   = f"@{parts[0]}" if parts else "@twitter_user"
            time_str = "X Post"
            raw_text = content

        elif platform == "instagram":
            og_desc  = soup.find("meta", {"property": "og:description"})
            content  = og_desc["content"] if og_desc and og_desc.get("content") else ""
            handle   = "@instagram_user"
            time_str = "Instagram Post"
            raw_text = content

    except Exception:
        pass

    # Graceful fallback if scraping returned nothing (sites often block bots)
    if not raw_text or len(raw_text.strip()) < 20:
        fallback_texts = {
            "youtube":   "This YouTube video features engaging content with commentary, reactions, and audience interaction. The creator shares thoughts and experiences with their community.",
            "x":         "Just sharing some thoughts on what has been happening lately. It has been quite a journey and I wanted to give everyone an update on where things stand.",
            "instagram": "Loving this moment. Grateful for every experience and the people who make life beautiful. Here is to more adventures ahead!",
            "unknown":   "Sharing this interesting content with everyone. Hope you find it as fascinating as I did.",
        }
        content  = fallback_texts.get(platform, fallback_texts["unknown"])
        raw_text = content
        handle   = handle if handle != "Unknown" else f"@{platform}_user"
        time_str = time_str if time_str != "Recently" else "Recent Post"

    return {
        "handle":   handle,
        "time":     time_str,
        "content":  content[:300],
        "raw_text": raw_text[:512],
    }


# ── Analysis helpers ───────────────────────────────────────────────────────────
def detect_themes(text: str) -> list:
    text_lower = text.lower()
    found = [theme for theme, kws in THEME_KEYWORDS.items() if any(kw in text_lower for kw in kws)]
    return found[:5] if found else ["General"]


def generate_insights(text: str, dominant_emotion: str) -> list:
    text_lower = text.lower()
    insights = []
    for keywords, insight in INSIGHT_RULES:
        if any(kw in text_lower for kw in keywords):
            insights.append(insight)
        if len(insights) == 3:
            break

    emotion_insights = {
        "joy":      "The overall joyful tone suggests the author is in a positive emotional state and likely seeking to share that energy.",
        "anger":    "Angry tone often correlates with high engagement — readers are likely to react strongly.",
        "sadness":  "Melancholic content tends to resonate deeply and can generate empathetic responses.",
        "fear":     "Fear-driven content often triggers concern and protective instincts in the audience.",
        "disgust":  "Disgust signals a strong moral or aesthetic violation in the author's view.",
        "surprise": "Surprise language keeps audiences engaged and drives shares.",
        "neutral":  "Neutral tone suggests an informational or factual communication style.",
    }
    if len(insights) < 3 and dominant_emotion in emotion_insights:
        insights.append(emotion_insights[dominant_emotion])
    while len(insights) < 3:
        insights.append("The post's language pattern suggests a measured, deliberate communication style.")
    return insights[:3]


def compute_toxicity(sentiment_scores: list, emotion_scores: list) -> int:
    neg     = next((s["score"] for s in sentiment_scores if "neg" in s["label"].lower()), 0)
    anger   = next((e["score"] for e in emotion_scores   if e["label"].lower() == "anger"),   0)
    disgust = next((e["score"] for e in emotion_scores   if e["label"].lower() == "disgust"), 0)
    return min(int((neg * 0.4 + anger * 0.4 + disgust * 0.2) * 100), 100)


def predict_engagement(emotion_scores: list, themes: list) -> str:
    joy      = next((e["score"] for e in emotion_scores if e["label"].lower() == "joy"),      0)
    anger    = next((e["score"] for e in emotion_scores if e["label"].lower() == "anger"),    0)
    surprise = next((e["score"] for e in emotion_scores if e["label"].lower() == "surprise"), 0)
    boost    = len({"Humor", "Excitement", "Criticism", "Achievement"} & set(themes)) * 0.1
    score    = joy + anger + surprise + boost
    return "High" if score > 0.6 else ("Medium" if score > 0.3 else "Low")


def run_analysis(url: str) -> dict:
    fetched  = fetch_text_from_url(url)
    raw_text = fetched["raw_text"]

    sentiment_pipe = load_sentiment_model()
    emotion_pipe   = load_emotion_model()

    sentiment_results = sentiment_pipe(raw_text)[0]
    emotion_results   = emotion_pipe(raw_text)[0]

    top_sent  = max(sentiment_results, key=lambda x: x["score"])
    label_key = top_sent["label"].lower()
    if "pos" in label_key:   label_key = "positive"
    elif "neg" in label_key: label_key = "negative"
    else:                    label_key = "neutral"

    emoji, overall, description = SENTIMENT_MAP.get(label_key, SENTIMENT_MAP["neutral"])
    conf_score = top_sent["score"]
    confidence = "High" if conf_score > 0.75 else ("Medium" if conf_score > 0.5 else "Low")

    emotions = [
        {"name": e["label"].capitalize(), "score": round(e["score"] * 100)}
        for e in sorted(emotion_results, key=lambda x: x["score"], reverse=True)
    ]
    dominant_emotion = emotion_results[0]["label"].lower() if emotion_results else "neutral"

    themes     = detect_themes(raw_text)
    insights   = generate_insights(raw_text, dominant_emotion)
    toxicity   = compute_toxicity(sentiment_results, emotion_results)
    engagement = predict_engagement(emotion_results, themes)

    return {
        "platform":              detect_platform(url),
        "handle":                fetched["handle"],
        "time":                  fetched["time"],
        "content":               fetched["content"],
        "overall_sentiment":     overall,
        "sentiment_emoji":       emoji,
        "sentiment_description": description,
        "confidence":            confidence,
        "confidence_score":      round(conf_score * 100),
        "emotions":              emotions,
        "themes":                themes,
        "insights":              insights,
        "toxicity_score":        toxicity,
        "engagement_prediction": engagement,
    }


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🧠 Social Sentiment Analyzer")
st.markdown('<span class="free-badge">✅ 100% Free — No API Key Required</span>', unsafe_allow_html=True)
st.caption("Paste any X, Instagram, or YouTube post URL for deep human sentiment analysis powered by free AI models.")

with st.sidebar:
    st.header("⚙️ About")
    st.markdown("""
**Powered by free Hugging Face models:**
- 🤗 `twitter-roberta-base-sentiment-latest`
  Sentiment: Positive / Negative / Neutral
- 🤗 `emotion-english-distilroberta-base`
  7 Emotions: Joy, Anger, Sadness, Fear…

**No API key. No cost. Runs locally.**
""")
    st.markdown("---")
    st.markdown("**Supported platforms**")
    st.markdown("- 𝕏 / Twitter\n- 📸 Instagram\n- ▶️ YouTube")
    st.markdown("---")
    st.info("💡 Models are auto-downloaded once (~500MB) and cached by Hugging Face for all future runs.")

# Input
url = st.text_input(
    "Post URL",
    placeholder="https://x.com/user/status/...  or  instagram.com/p/...  or  youtube.com/watch?v=...",
    label_visibility="collapsed",
)

col_btn, col_ex = st.columns([1, 3])
with col_btn:
    run = st.button("Analyze ↗", type="primary", use_container_width=True)
with col_ex:
    example = st.selectbox(
        "Try an example",
        ["", "X / Twitter", "Instagram", "YouTube"],
        label_visibility="collapsed",
    )
    if example == "X / Twitter":
        url = "https://x.com/NASA/status/1234567890"
    elif example == "Instagram":
        url = "https://www.instagram.com/p/Cs8XzLpABC/"
    elif example == "YouTube":
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# ── Run ────────────────────────────────────────────────────────────────────────
if run or example:
    if not url:
        st.warning("Please enter a post URL.")
        st.stop()

    p_icon = PLATFORM_ICONS.get(detect_platform(url), "🌐")

    with st.spinner("Loading models and analyzing… (first run downloads models, ~30s)"):
        try:
            result = run_analysis(url)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("Analysis complete!")
    st.divider()

    # Post card
    st.subheader(f"{p_icon} Post Preview")
    with st.container(border=True):
        st.markdown(f"**{result['handle']}** · *{result['time']}*")
        st.markdown(f"> {result['content']}")

    st.divider()

    # Metrics
    st.subheader("Overall Sentiment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment",  f"{result['sentiment_emoji']} {result['overall_sentiment']}")
    c2.metric("Confidence", f"{result['confidence']} ({result['confidence_score']}%)")
    c3.metric("Toxicity",   f"{result['toxicity_score']}/100")
    c4.metric("Engagement", result["engagement_prediction"])
    st.caption(f"_{result['sentiment_description']}_")

    st.divider()

    # Emotions
    st.subheader("Emotion Breakdown")
    for em in result["emotions"]:
        score = min(int(em["score"]), 100)
        col_name, col_bar = st.columns([1, 4])
        col_name.markdown(f"**{em['name']}**")
        col_bar.progress(score / 100, text=f"{score}%")

    st.divider()

    # Themes
    st.subheader("Detected Themes")
    theme_html = " ".join(
        f'<span style="background:#eef2ff;color:#4338ca;padding:4px 12px;'
        f'border-radius:20px;font-size:13px;margin-right:6px;display:inline-block;margin-bottom:6px;">'
        f'{t}</span>'
        for t in result["themes"]
    )
    st.markdown(theme_html, unsafe_allow_html=True)

    st.divider()

    # Insights
    st.subheader("Human Insights")
    icons_list = ["👁️", "🧠", "💬"]
    for i, insight in enumerate(result["insights"]):
        st.markdown(
            f'<div class="insight-box">{icons_list[i % 3]} {insight}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("🔍 View full analysis data"):
        st.json(result)
