import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    font-size: 13.5px;
    border-radius: 8px;
}
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
.stProgress > div > div > div { border-radius: 4px; }

.insight-box {
    background: #f8f9fa;
    border-left: 3px solid #6c63ff;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #333;
}
.platform-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}
.step-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #92400e;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_API_URL_SENTIMENT = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_API_URL_EMOTION   = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

PLATFORM_ICONS  = {"x": "𝕏", "instagram": "📸", "youtube": "▶️", "unknown": "🌐"}
PLATFORM_COLORS = {"x": "#000000", "instagram": "#c13584", "youtube": "#cc0000", "unknown": "#666"}

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
    "News":        ["breaking", "update", "report", "announced", "today", "just in"],
    "Gratitude":   ["thank", "grateful", "appreciate", "blessed", "thankful"],
    "Question":    ["?", "why", "how", "what", "when", "who", "wonder"],
    "Nostalgia":   ["remember", "miss", "used to", "back then", "old days", "throwback"],
    "Excitement":  ["amazing", "wow", "omg", "excited", "can't wait", "thrilled"],
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
     "Absolute language ('always/never') may indicate strong conviction or emotional reactivity."),
    (["lol", "haha", "joke", "funny"],
     "Humor markers suggest levity, possibly used to soften a serious point."),
    (["miss", "remember", "used to", "back", "throwback"],
     "Nostalgic phrasing reflects an emotional attachment to the past."),
    (["please", "help", "need", "support"],
     "Appeal language ('please/help/need') signals vulnerability or a direct call to action."),
]

EMOTION_INSIGHTS = {
    "joy":      "The joyful tone suggests the author is in a positive emotional state seeking to share that energy.",
    "anger":    "Angry tone often correlates with high engagement — readers are likely to react strongly.",
    "sadness":  "Melancholic content tends to resonate deeply and generates empathetic responses.",
    "fear":     "Fear-driven content triggers concern and protective instincts in the audience.",
    "disgust":  "Disgust signals a strong moral or aesthetic violation from the author's perspective.",
    "surprise": "Surprise language keeps audiences engaged and drives organic sharing.",
    "neutral":  "Neutral tone suggests an informational or factual communication style.",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def detect_platform(url: str) -> str:
    u = url.lower()
    if "x.com" in u or "twitter.com" in u:   return "x"
    if "instagram.com" in u:                  return "instagram"
    if "youtube.com" in u or "youtu.be" in u: return "youtube"
    return "unknown"


def fetch_youtube_meta(url: str) -> dict:
    """YouTube publicly exposes og: meta tags — reliable scraping."""
    headers = {"User-Agent": "Mozilla/5.0"}
    handle, time_str, content = "YouTube Creator", "YouTube Video", ""
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag   = soup.find("meta", {"property": "og:title"})
        desc_tag    = soup.find("meta", {"property": "og:description"})
        channel_tag = soup.find("link", {"itemprop": "name"})
        title   = title_tag["content"]   if title_tag   and title_tag.get("content")   else ""
        desc    = desc_tag["content"]    if desc_tag    and desc_tag.get("content")    else ""
        handle  = channel_tag["content"] if channel_tag and channel_tag.get("content") else "YouTube Creator"
        content = f"{title}. {desc}".strip(". ") if title or desc else ""
    except Exception:
        pass
    return {"handle": handle, "time": time_str, "content": content}


def hf_infer(api_url: str, text: str, hf_token: str = "") -> list:
    """
    Call Hugging Face Inference API.
    Works without a token for public models (rate-limited to ~30 req/hr).
    Pass a free HF token to raise the limit.
    Retries once if the model is loading (503).
    """
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {"inputs": text, "parameters": {"return_all_scores": True}}

    for attempt in range(3):
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 503:          # model warming up
            time.sleep(10)
            continue
        if resp.status_code == 200:
            data = resp.json()
            # API returns [[{label, score}, ...]] or [{label, score}, ...]
            if isinstance(data, list) and len(data) > 0:
                inner = data[0] if isinstance(data[0], list) else data
                return inner
        break
    return []


def detect_themes(text: str) -> list:
    tl = text.lower()
    found = [t for t, kws in THEME_KEYWORDS.items() if any(k in tl for k in kws)]
    return found[:5] or ["General"]


def generate_insights(text: str, dominant_emotion: str) -> list:
    tl = text.lower()
    insights = []
    for keywords, insight in INSIGHT_RULES:
        if any(k in tl for k in keywords):
            insights.append(insight)
        if len(insights) == 3:
            break
    if len(insights) < 3 and dominant_emotion in EMOTION_INSIGHTS:
        insights.append(EMOTION_INSIGHTS[dominant_emotion])
    while len(insights) < 3:
        insights.append("The language pattern suggests a measured, deliberate communication style.")
    return insights[:3]


def compute_toxicity(sent_scores: list, emo_scores: list) -> int:
    neg     = next((s["score"] for s in sent_scores if "neg" in s["label"].lower()), 0)
    anger   = next((e["score"] for e in emo_scores  if e["label"].lower() == "anger"),   0)
    disgust = next((e["score"] for e in emo_scores  if e["label"].lower() == "disgust"), 0)
    return min(int((neg * 0.4 + anger * 0.4 + disgust * 0.2) * 100), 100)


def predict_engagement(emo_scores: list, themes: list) -> str:
    joy      = next((e["score"] for e in emo_scores if e["label"].lower() == "joy"),      0)
    anger    = next((e["score"] for e in emo_scores if e["label"].lower() == "anger"),    0)
    surprise = next((e["score"] for e in emo_scores if e["label"].lower() == "surprise"), 0)
    boost    = len({"Humor", "Excitement", "Criticism", "Achievement"} & set(themes)) * 0.1
    score    = joy + anger + surprise + boost
    return "High" if score > 0.6 else ("Medium" if score > 0.3 else "Low")


def run_analysis(url: str, post_text: str, hf_token: str) -> dict:
    platform = detect_platform(url)
    handle, time_str, content = f"@{platform}_user", f"{platform.capitalize()} Post", post_text

    # For YouTube, try to auto-fetch title+description
    if platform == "youtube" and not post_text.strip():
        meta    = fetch_youtube_meta(url)
        handle  = meta["handle"]
        time_str = meta["time"]
        content = meta["content"] or "A YouTube video with commentary and community engagement."

    if not content.strip():
        content = post_text.strip() or "No content could be extracted from this post."

    # Pull handle from X URL path
    if platform == "x":
        parts  = urlparse(url).path.strip("/").split("/")
        handle = f"@{parts[0]}" if parts and parts[0] else "@x_user"

    raw_text = content[:512]

    # ── HF Inference API calls ─────────────────────────────────────────────
    sent_scores = hf_infer(HF_API_URL_SENTIMENT, raw_text, hf_token)
    emo_scores  = hf_infer(HF_API_URL_EMOTION,   raw_text, hf_token)

    if not sent_scores:
        raise RuntimeError("Sentiment model did not respond. The HF Inference API may be rate-limited — try again in a minute, or add a free HF token in the sidebar.")
    if not emo_scores:
        raise RuntimeError("Emotion model did not respond. Try again in a moment.")

    # Normalise sentiment label
    top_sent  = max(sent_scores, key=lambda x: x["score"])
    lbl       = top_sent["label"].lower()
    lbl_key   = "positive" if "pos" in lbl else ("negative" if "neg" in lbl else "neutral")
    emoji_s, overall, description = SENTIMENT_MAP[lbl_key]
    conf_score = top_sent["score"]
    confidence = "High" if conf_score > 0.75 else ("Medium" if conf_score > 0.5 else "Low")

    # Emotions sorted
    emotions = [
        {"name": e["label"].capitalize(), "score": round(e["score"] * 100)}
        for e in sorted(emo_scores, key=lambda x: x["score"], reverse=True)
    ]
    dominant_emotion = emo_scores[0]["label"].lower() if emo_scores else "neutral"

    themes     = detect_themes(raw_text)
    insights   = generate_insights(raw_text, dominant_emotion)
    toxicity   = compute_toxicity(sent_scores, emo_scores)
    engagement = predict_engagement(emo_scores, themes)

    return {
        "platform":              platform,
        "handle":                handle,
        "time":                  time_str,
        "content":               content[:300],
        "overall_sentiment":     overall,
        "sentiment_emoji":       emoji_s,
        "sentiment_description": description,
        "confidence":            confidence,
        "confidence_score":      round(conf_score * 100),
        "emotions":              emotions,
        "themes":                themes,
        "insights":              insights,
        "toxicity_score":        toxicity,
        "engagement_prediction": engagement,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Social Sentiment Analyzer")
st.caption("Analyze human emotion and sentiment from X, Instagram, or YouTube posts — no API key required.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    hf_token = st.text_input(
        "HF Token (optional)",
        type="password",
        placeholder="hf_...",
        help=(
            "Optional free Hugging Face token to avoid rate limits. "
            "Get one free at huggingface.co/settings/tokens"
        ),
    )

    st.markdown("---")
    st.markdown("**🤗 Powered by free HF models**")
    st.markdown(
        "- `twitter-roberta-base-sentiment` \n"
        "  → Positive / Negative / Neutral\n"
        "- `emotion-english-distilroberta` \n"
        "  → Joy, Anger, Sadness, Fear…"
    )
    st.markdown("---")
    st.markdown("**Supported platforms**")
    st.markdown("- 𝕏 / Twitter\n- 📸 Instagram\n- ▶️ YouTube")
    st.markdown("---")
    st.info(
        "**No heavy libraries.** Runs via lightweight HTTP calls to "
        "Hugging Face Inference API — works perfectly on Streamlit Cloud."
    )

# ── Step 1: URL ────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Paste the post URL")
url = st.text_input(
    "Post URL",
    placeholder="https://x.com/user/status/...   |   instagram.com/p/...   |   youtube.com/watch?v=...",
    label_visibility="collapsed",
)

# Quick-fill examples
col_x, col_ig, col_yt = st.columns(3)
if col_x.button("𝕏  Example",  use_container_width=True):
    url = "https://x.com/NASA/status/1234567890"
if col_ig.button("📸 Example", use_container_width=True):
    url = "https://www.instagram.com/p/Cs8XzLpABC/"
if col_yt.button("▶️ Example", use_container_width=True):
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Platform detection preview
if url:
    platform_detected = detect_platform(url)
    color = PLATFORM_COLORS.get(platform_detected, "#666")
    icon  = PLATFORM_ICONS.get(platform_detected, "🌐")
    st.markdown(
        f'<span class="platform-pill" style="background:{color}22;color:{color};">'
        f'{icon} {platform_detected.upper()} detected</span>',
        unsafe_allow_html=True,
    )

# ── Step 2: Post text ──────────────────────────────────────────────────────────
st.subheader("Step 2 — Paste the post text")

st.markdown(
    '<div class="step-box">'
    '⚠️ <strong>X and Instagram block automated scraping.</strong> '
    'Open the post in your browser, copy the caption / tweet text, and paste it below. '
    'For YouTube, this field is optional — the title & description are fetched automatically.'
    '</div>',
    unsafe_allow_html=True,
)

post_text = st.text_area(
    "Post text",
    placeholder=(
        "Paste the tweet, caption, or video description here…\n\n"
        "Example: \"Just launched our new product! Incredibly proud of the whole team 🎉 "
        "This wouldn't have been possible without everyone's support. Thank you!\""
    ),
    height=130,
    label_visibility="collapsed",
)

# ── Analyze button ─────────────────────────────────────────────────────────────
st.markdown("")
run = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════
if run:
    if not url:
        st.warning("Please enter a post URL in Step 1.")
        st.stop()

    platform = detect_platform(url)

    if platform in ("x", "instagram") and not post_text.strip():
        st.warning(
            f"Please paste the post text in Step 2. "
            f"{'X' if platform == 'x' else 'Instagram'} blocks automated scraping, "
            f"so the text cannot be fetched automatically."
        )
        st.stop()

    with st.spinner("Calling AI models… this takes about 5–10 seconds ⏳"):
        try:
            result = run_analysis(url, post_text, hf_token)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.success("✅ Analysis complete!")
    st.divider()

    # ── Post card ──────────────────────────────────────────────────────────────
    p_icon = PLATFORM_ICONS.get(result["platform"], "🌐")
    st.subheader(f"{p_icon} Post Preview")
    with st.container(border=True):
        st.markdown(f"**{result['handle']}** · *{result['time']}*")
        st.markdown(f"> {result['content']}")

    st.divider()

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.subheader("Overall Sentiment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment",  f"{result['sentiment_emoji']} {result['overall_sentiment']}")
    c2.metric("Confidence", f"{result['confidence']} ({result['confidence_score']}%)")
    c3.metric("Toxicity",   f"{result['toxicity_score']}/100")
    c4.metric("Engagement", result["engagement_prediction"])
    st.caption(f"_{result['sentiment_description']}_")

    st.divider()

    # ── Emotion breakdown ──────────────────────────────────────────────────────
    st.subheader("Emotion Breakdown")
    for em in result["emotions"]:
        score = min(int(em["score"]), 100)
        cn, cb = st.columns([1, 4])
        cn.markdown(f"**{em['name']}**")
        cb.progress(score / 100, text=f"{score}%")

    st.divider()

    # ── Themes ─────────────────────────────────────────────────────────────────
    st.subheader("Detected Themes")
    theme_html = " ".join(
        f'<span style="background:#eef2ff;color:#4338ca;padding:4px 12px;border-radius:20px;'
        f'font-size:13px;margin-right:6px;display:inline-block;margin-bottom:6px;">{t}</span>'
        for t in result["themes"]
    )
    st.markdown(theme_html, unsafe_allow_html=True)

    st.divider()

    # ── Human insights ─────────────────────────────────────────────────────────
    st.subheader("Human Insights")
    icons_list = ["👁️", "🧠", "💬"]
    for i, insight in enumerate(result["insights"]):
        st.markdown(
            f'<div class="insight-box">{icons_list[i % 3]} {insight}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("🔍 View raw analysis data"):
        st.json(result)
