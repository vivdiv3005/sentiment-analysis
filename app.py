import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

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
.free-badge {
    display: inline-block;
    background: #d1fae5;
    color: #065f46;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PLATFORM_ICONS  = {"x": "𝕏", "instagram": "📸", "youtube": "▶️", "unknown": "🌐"}
PLATFORM_COLORS = {"x": "#000000", "instagram": "#c13584", "youtube": "#cc0000", "unknown": "#555"}

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
     "Direct address suggests the author is speaking to or challenging the audience."),
    (["but", "however", "although", "yet", "despite"],
     "Contrasting language signals nuanced thinking or emotional ambivalence."),
    (["always", "never", "everyone", "nobody", "all"],
     "Absolute language may indicate strong conviction or emotional reactivity."),
    (["lol", "haha", "joke", "funny"],
     "Humor markers suggest levity, possibly used to soften a serious point."),
    (["miss", "remember", "used to", "back", "throwback"],
     "Nostalgic phrasing reflects an emotional attachment to the past."),
    (["please", "help", "need", "support"],
     "Appeal language signals vulnerability or a direct call to action."),
]


# ── Platform detection ─────────────────────────────────────────────────────────
def detect_platform(url: str) -> str:
    u = url.lower()
    if "x.com" in u or "twitter.com" in u:   return "x"
    if "instagram.com" in u:                  return "instagram"
    if "youtube.com" in u or "youtu.be" in u: return "youtube"
    return "unknown"


# ── YouTube meta fetch ─────────────────────────────────────────────────────────
def fetch_youtube_meta(url: str) -> dict:
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
        content = f"{title}. {desc}".strip(". ")
    except Exception:
        pass
    return {"handle": handle, "time": time_str, "content": content}


# ── Core analysis (fully local, no API) ───────────────────────────────────────
def analyze_text(text: str) -> dict:
    """
    Runs two complementary local analyzers:
      • VADER  — rule-based, great for social media slang, emojis, CAPS
      • TextBlob — pattern-based, gives polarity + subjectivity
    Combines both for a richer result.
    """
    analyzer = SentimentIntensityAnalyzer()
    vader    = analyzer.polarity_scores(text)   # {neg, neu, pos, compound}
    blob     = TextBlob(text)
    polarity     = blob.sentiment.polarity       # -1 to +1
    subjectivity = blob.sentiment.subjectivity   # 0 (objective) to 1 (subjective)

    # ── Overall sentiment from VADER compound (more robust for social text) ──
    compound = vader["compound"]
    if compound >= 0.05:
        sentiment_key = "positive"
        sentiment_label = "Positive"
        sentiment_emoji = "😄"
        sentiment_desc  = "The content carries an overall positive and uplifting tone."
    elif compound <= -0.05:
        sentiment_key = "negative"
        sentiment_label = "Negative"
        sentiment_emoji = "😞"
        sentiment_desc  = "The content carries a negative or critical emotional tone."
    else:
        sentiment_key = "neutral"
        sentiment_label = "Neutral"
        sentiment_emoji = "😐"
        sentiment_desc  = "The content is balanced or matter-of-fact in tone."

    # ── Confidence: average agreement between VADER and TextBlob ─────────────
    vader_score  = (compound + 1) / 2           # map -1..1 → 0..1
    blob_score   = (polarity + 1) / 2
    agreement    = 1 - abs(vader_score - blob_score)
    conf_pct     = int(round(agreement * 100))
    confidence   = "High" if conf_pct >= 70 else ("Medium" if conf_pct >= 45 else "Low")

    # ── Pseudo-emotion scores derived from VADER + linguistic cues ───────────
    tl = text.lower()
    joy_boost      = 0.2 if any(w in tl for w in ["happy","love","great","amazing","yay","🎉","😍"]) else 0
    anger_boost    = 0.2 if any(w in tl for w in ["hate","angry","furious","!!","wtf","disgusting"])  else 0
    surprise_boost = 0.15 if any(w in tl for w in ["wow","omg","unbelievable","shocked","what?!"])    else 0
    nostalgia_boost= 0.1  if any(w in tl for w in ["miss","remember","throwback","used to","back when"]) else 0

    joy       = min(round((vader["pos"] + joy_boost) * 100), 100)
    sadness   = min(round((vader["neg"] * 0.6 + max(-polarity, 0) * 0.4) * 100), 100)
    anger     = min(round((vader["neg"] * 0.5 + anger_boost) * 100), 100)
    fear      = min(round((vader["neg"] * 0.3) * 100), 100)
    surprise  = min(round((abs(compound) * 0.3 + surprise_boost) * 100), 100)
    neutral_e = min(round(vader["neu"] * 100), 100)
    disgust   = min(round((vader["neg"] * 0.4) * 100), 100)

    emotions = sorted([
        {"name": "Joy",      "score": joy},
        {"name": "Sadness",  "score": sadness},
        {"name": "Anger",    "score": anger},
        {"name": "Fear",     "score": fear},
        {"name": "Surprise", "score": surprise},
        {"name": "Neutral",  "score": neutral_e},
        {"name": "Disgust",  "score": disgust},
    ], key=lambda e: e["score"], reverse=True)

    dominant_emotion = emotions[0]["name"].lower()

    # ── Toxicity ──────────────────────────────────────────────────────────────
    toxicity = min(int((vader["neg"] * 0.5 + anger * 0.003 + disgust * 0.002) * 100), 100)

    # ── Subjectivity label ────────────────────────────────────────────────────
    if subjectivity >= 0.6:
        subjectivity_label = "Highly Subjective"
    elif subjectivity >= 0.35:
        subjectivity_label = "Moderately Subjective"
    else:
        subjectivity_label = "Mostly Objective"

    # ── Engagement prediction ─────────────────────────────────────────────────
    excl_count = text.count("!") + text.count("?")
    eng_score  = abs(compound) + (excl_count * 0.05) + (joy_boost + anger_boost + surprise_boost)
    engagement = "High" if eng_score > 0.65 else ("Medium" if eng_score > 0.3 else "Low")

    # ── Themes ────────────────────────────────────────────────────────────────
    themes = [t for t, kws in THEME_KEYWORDS.items() if any(k in tl for k in kws)]
    themes = themes[:5] or ["General"]

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = []
    for keywords, insight in INSIGHT_RULES:
        if any(k in tl for k in keywords):
            insights.append(insight)
        if len(insights) == 3:
            break

    emotion_insights = {
        "joy":      "The joyful tone suggests the author is in a positive emotional state seeking to share that energy.",
        "anger":    "Angry tone often correlates with high engagement — readers are likely to react strongly.",
        "sadness":  "Melancholic content tends to resonate deeply and generates empathetic responses.",
        "fear":     "Fear-driven content triggers concern and protective instincts in the audience.",
        "disgust":  "Disgust signals a strong moral or aesthetic violation from the author's perspective.",
        "surprise": "Surprise language keeps audiences engaged and drives organic sharing.",
        "neutral":  "Neutral tone suggests an informational or factual communication style.",
    }
    if len(insights) < 3 and dominant_emotion in emotion_insights:
        insights.append(emotion_insights[dominant_emotion])
    while len(insights) < 3:
        insights.append("The language pattern suggests a measured, deliberate communication style.")

    return {
        "sentiment_label":       sentiment_label,
        "sentiment_emoji":       sentiment_emoji,
        "sentiment_description": sentiment_desc,
        "confidence":            confidence,
        "confidence_score":      conf_pct,
        "vader_compound":        round(compound, 3),
        "polarity":              round(polarity, 3),
        "subjectivity":          round(subjectivity, 3),
        "subjectivity_label":    subjectivity_label,
        "emotions":              emotions,
        "toxicity_score":        toxicity,
        "engagement_prediction": engagement,
        "themes":                themes,
        "insights":              insights[:3],
    }


def run_full_analysis(url: str, post_text: str) -> dict:
    platform  = detect_platform(url)
    handle    = f"@{platform}_user"
    time_str  = f"{platform.capitalize()} Post"
    content   = post_text.strip()

    if platform == "youtube" and not content:
        meta     = fetch_youtube_meta(url)
        handle   = meta["handle"]
        time_str = meta["time"]
        content  = meta["content"] or "A YouTube video with commentary and community engagement."

    if platform == "x":
        parts  = urlparse(url).path.strip("/").split("/")
        handle = f"@{parts[0]}" if parts and parts[0] else "@x_user"

    if not content:
        content = "No content provided."

    analysis = analyze_text(content[:512])

    return {
        "platform": platform,
        "handle":   handle,
        "time":     time_str,
        "content":  content[:300],
        **analysis,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Social Sentiment Analyzer")
st.markdown('<span class="free-badge">✅ 100% Free · No API Key · Works on Streamlit Cloud</span>', unsafe_allow_html=True)
st.caption("Analyze human emotion and sentiment from X, Instagram, or YouTube posts using local AI — instant results, no external API calls.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ About")
    st.markdown("""
**Powered by local libraries:**
- 🔵 **VADER** — rule-based, built for social media text, emojis & slang
- 🟣 **TextBlob** — pattern-based polarity & subjectivity scoring

**Why local?**
No cold starts, no rate limits, no API keys.
Works instantly every time.
""")
    st.markdown("---")
    st.markdown("**Supported platforms**")
    st.markdown("- 𝕏 / Twitter\n- 📸 Instagram\n- ▶️ YouTube")
    st.markdown("---")
    st.markdown("**Scores explained**")
    st.markdown(
        "- **VADER compound**: -1 (most negative) → +1 (most positive)\n"
        "- **Polarity**: TextBlob's -1 → +1 score\n"
        "- **Subjectivity**: 0 (factual) → 1 (opinionated)\n"
        "- **Toxicity**: 0 (clean) → 100 (very toxic)"
    )

# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Paste the post URL")
url = st.text_input(
    "url",
    placeholder="https://x.com/user/status/...   |   instagram.com/p/...   |   youtube.com/watch?v=...",
    label_visibility="collapsed",
)

col_x, col_ig, col_yt = st.columns(3)
if col_x.button("𝕏 X Example",       use_container_width=True):
    url = "https://x.com/NASA/status/1234567890"
if col_ig.button("📸 Instagram Eg",   use_container_width=True):
    url = "https://www.instagram.com/p/Cs8XzLpABC/"
if col_yt.button("▶️ YouTube Eg",     use_container_width=True):
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

if url:
    pl     = detect_platform(url)
    color  = PLATFORM_COLORS.get(pl, "#555")
    icon   = PLATFORM_ICONS.get(pl, "🌐")
    st.markdown(
        f'<span class="platform-pill" style="background:{color}22;color:{color};">'
        f'{icon} {pl.upper()} detected</span>',
        unsafe_allow_html=True,
    )

# ── Step 2 ─────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Paste the post text")
st.markdown(
    '<div class="step-box">'
    '⚠️ <b>X and Instagram block scraping.</b> '
    'Open the post in your browser, copy the text, and paste it below. '
    'For <b>YouTube</b> this is optional — title & description are auto-fetched.'
    '</div>',
    unsafe_allow_html=True,
)
post_text = st.text_area(
    "post_text",
    placeholder=(
        'Paste the tweet, caption, or video description here…\n\n'
        'Example: "Just launched our new product! Incredibly proud of the whole team 🎉 '
        'This wouldn\'t have been possible without everyone\'s support. Thank you!"'
    ),
    height=130,
    label_visibility="collapsed",
)

st.markdown("")
run = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run:
    if not url:
        st.warning("Please enter a post URL in Step 1.")
        st.stop()

    pl = detect_platform(url)
    if pl in ("x", "instagram") and not post_text.strip():
        st.warning(
            f"Please paste the post text in Step 2. "
            f"{'X' if pl == 'x' else 'Instagram'} blocks automated scraping."
        )
        st.stop()

    with st.spinner("Analyzing…"):
        try:
            result = run_full_analysis(url, post_text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("✅ Analysis complete!")
    st.divider()

    # Post card
    p_icon = PLATFORM_ICONS.get(result["platform"], "🌐")
    st.subheader(f"{p_icon} Post Preview")
    with st.container(border=True):
        st.markdown(f"**{result['handle']}** · *{result['time']}*")
        st.markdown(f"> {result['content']}")

    st.divider()

    # Metrics row 1
    st.subheader("Overall Sentiment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment",   f"{result['sentiment_emoji']} {result['sentiment_label']}")
    c2.metric("Confidence",  f"{result['confidence']} ({result['confidence_score']}%)")
    c3.metric("Toxicity",    f"{result['toxicity_score']}/100")
    c4.metric("Engagement",  result["engagement_prediction"])
    st.caption(f"_{result['sentiment_description']}_")

    # Metrics row 2 — detailed scores
    st.markdown("")
    d1, d2, d3 = st.columns(3)
    d1.metric("VADER Compound",  result["vader_compound"],
              help="−1 = most negative, +1 = most positive")
    d2.metric("TextBlob Polarity", result["polarity"],
              help="−1 = negative, +1 = positive")
    d3.metric("Subjectivity",    f"{result['subjectivity']} — {result['subjectivity_label']}",
              help="0 = objective, 1 = subjective")

    st.divider()

    # Emotion breakdown
    st.subheader("Emotion Breakdown")
    for em in result["emotions"]:
        score = min(int(em["score"]), 100)
        cn, cb = st.columns([1, 4])
        cn.markdown(f"**{em['name']}**")
        cb.progress(score / 100, text=f"{score}%")

    st.divider()

    # Themes
    st.subheader("Detected Themes")
    theme_html = " ".join(
        f'<span style="background:#eef2ff;color:#4338ca;padding:4px 12px;border-radius:20px;'
        f'font-size:13px;margin-right:6px;display:inline-block;margin-bottom:6px;">{t}</span>'
        for t in result["themes"]
    )
    st.markdown(theme_html, unsafe_allow_html=True)

    st.divider()

    # Human insights
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
