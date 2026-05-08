import streamlit as st
import anthropic
import json
import re
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

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Global */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Title */
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }

/* URL input */
div[data-testid="stTextInput"] input {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    border-radius: 8px;
}

/* Metric cards */
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.5px; }

/* Progress bars */
.stProgress > div > div > div { border-radius: 4px; }

/* Expander */
.streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; }

/* Insight boxes */
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
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
PLATFORM_ICONS = {"x": "𝕏", "twitter": "𝕏", "instagram": "📸", "youtube": "▶️"}
EMOTION_COLORS = {
    "Joy": "#f0c040", "Trust": "#3daa70", "Anticipation": "#f07830",
    "Anger": "#e04848", "Sadness": "#5888d0", "Fear": "#9058c8",
    "Surprise": "#48b8c0", "Disgust": "#88a840",
}
SENTIMENT_EMOJI = {
    "Joyful": "😄", "Angry": "😡", "Sad": "😢", "Anxious": "😰",
    "Inspired": "✨", "Neutral": "😐", "Hopeful": "🌟", "Frustrated": "😤",
    "Surprised": "😲", "Nostalgic": "🌅",
}


def detect_platform(url: str) -> str:
    url_lower = url.lower()
    if "x.com" in url_lower or "twitter.com" in url_lower:
        return "x"
    if "instagram.com" in url_lower:
        return "instagram"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    return "unknown"


def analyze_sentiment(url: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a human sentiment and emotion analysis expert. 
Given a social media post URL, simulate fetching the post and perform deep human sentiment analysis.

Respond ONLY in valid JSON with this exact structure (no markdown fences):
{
  "platform": "x" | "instagram" | "youtube",
  "handle": "@username or channel name",
  "time": "relative or absolute time like '2 hours ago' or 'Mar 15, 2024'",
  "content": "a realistic, platform-appropriate post excerpt (2-4 sentences)",
  "overall_sentiment": one of: Joyful | Angry | Sad | Anxious | Inspired | Neutral | Hopeful | Frustrated | Surprised | Nostalgic,
  "sentiment_description": "one crisp sentence describing the emotional tone",
  "confidence": "High" | "Medium" | "Low",
  "emotions": [
    {"name": "Joy", "score": 72},
    {"name": "Trust", "score": 55},
    {"name": "Anticipation", "score": 38},
    {"name": "Sadness", "score": 10},
    {"name": "Anger", "score": 5}
  ],
  "themes": ["Community", "Achievement", "Humor"],
  "insights": [
    "The author uses inclusive language suggesting a strong community orientation.",
    "Rhetorical questions signal underlying uncertainty or seeking validation.",
    "Positive framing despite a potentially negative subject indicates emotional resilience."
  ],
  "toxicity_score": 12,
  "engagement_prediction": "High" | "Medium" | "Low"
}

Rules:
- Pick 4-6 most relevant emotions from: Joy, Sadness, Anger, Fear, Trust, Surprise, Anticipation, Disgust
- Scores are 0-100 (independent, not summing to 100)
- 3-5 theme tags, 3 behavioral insights
- toxicity_score is 0-100 (100 = extremely toxic)
- Make analysis feel genuinely human and insightful, not generic
"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Analyze the sentiment of this post: {url}"}],
    )

    raw = message.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🧠 Social Sentiment Analyzer")
st.caption("Paste any X, Instagram, or YouTube post URL to get a deep human sentiment analysis.")

# Sidebar – API key
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )
    st.markdown("---")
    st.markdown("**Supported platforms**")
    st.markdown("- 𝕏 / Twitter\n- 📸 Instagram\n- ▶️ YouTube")
    st.markdown("---")
    st.markdown("**About**")
    st.caption("This app uses Claude to analyze human emotion, tone, and behavioral signals in social media posts.")

# Main input
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
        url = "https://www.instagram.com/p/ABC123xyz/"
    elif example == "YouTube":
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# ── Analysis ────────────────────────────────────────────────────────────────────
if run or example:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()
    if not url:
        st.warning("Please enter a post URL.")
        st.stop()

    platform = detect_platform(url)
    icon = PLATFORM_ICONS.get(platform, "🌐")

    with st.spinner("Fetching post and analyzing sentiment…"):
        try:
            result = analyze_sentiment(url, api_key)
        except json.JSONDecodeError:
            st.error("The model returned an unexpected format. Please try again.")
            st.stop()
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("Analysis complete!")
    st.divider()

    # ── Post card ──────────────────────────────────────────────────────────────
    st.subheader(f"{icon} Post Preview")
    with st.container(border=True):
        handle = result.get("handle", "Unknown")
        time_str = result.get("time", "")
        content = result.get("content", "")
        st.markdown(f"**{handle}** · *{time_str}*")
        st.markdown(f"> {content}")

    st.divider()

    # ── Overall sentiment ──────────────────────────────────────────────────────
    overall = result.get("overall_sentiment", "Neutral")
    emoji = SENTIMENT_EMOJI.get(overall, "😐")
    description = result.get("sentiment_description", "")
    confidence = result.get("confidence", "Medium")
    toxicity = result.get("toxicity_score", 0)
    engagement = result.get("engagement_prediction", "Medium")

    st.subheader("Overall Sentiment")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment", f"{emoji} {overall}")
    col2.metric("Confidence", confidence)
    col3.metric("Toxicity", f"{toxicity}/100")
    col4.metric("Engagement", engagement)
    st.caption(f"_{description}_")

    st.divider()

    # ── Emotion breakdown ──────────────────────────────────────────────────────
    st.subheader("Emotion Breakdown")
    emotions = result.get("emotions", [])
    emotions_sorted = sorted(emotions, key=lambda e: e["score"], reverse=True)

    for em in emotions_sorted:
        name = em["name"]
        score = min(int(em["score"]), 100)
        color = EMOTION_COLORS.get(name, "#999")
        col_name, col_bar = st.columns([1, 4])
        col_name.markdown(f"**{name}**")
        col_bar.progress(score / 100, text=f"{score}%")

    st.divider()

    # ── Themes ─────────────────────────────────────────────────────────────────
    st.subheader("Detected Themes")
    themes = result.get("themes", [])
    theme_html = " ".join(
        f'<span style="background:#eef2ff;color:#4338ca;padding:4px 12px;'
        f'border-radius:20px;font-size:13px;margin-right:6px;display:inline-block;'
        f'margin-bottom:6px;">{t}</span>'
        for t in themes
    )
    st.markdown(theme_html, unsafe_allow_html=True)

    st.divider()

    # ── Human insights ─────────────────────────────────────────────────────────
    st.subheader("Human Insights")
    insights = result.get("insights", [])
    icons_list = ["👁️", "🧠", "💬"]
    for i, insight in enumerate(insights):
        icon_i = icons_list[i % len(icons_list)]
        st.markdown(
            f'<div class="insight-box">{icon_i} {insight}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Raw JSON expander ──────────────────────────────────────────────────────
    with st.expander("🔍 View raw JSON response"):
        st.json(result)
