import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shift_detector import detect_shifts

st.set_page_config(page_title="Hinglish Sentiment Shift Detector", layout="wide", initial_sidebar_state="collapsed")

HISTORY_FILE = "outputs/results/history.json"

def save_history(history):
    os.makedirs("outputs/results", exist_ok=True)
    saveable = []
    for item in history:
        saveable.append({
            'preview': item['preview'],
            'overall': item['overall'],
            'shifts': item['shifts'],
            'raw': item['raw']
        })
    with open(HISTORY_FILE, 'w') as f:
        json.dump(saveable, f)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_profile(name, college, branch, semester):
    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/profile.json", 'w') as f:
        json.dump({'name': name, 'college': college, 'branch': branch, 'semester': semester}, f)

def load_profile():
    path = "outputs/results/profile.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'name': 'Phoenix', 'college': 'JECRC University', 'branch': 'CSE', 'semester': 'VI Semester'}

for key, val in {
    'dark_mode': False, 'page': 'home', 'results': None,
    'history': None, 'conversation_text': "", 'show_options': False,
    'profile': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.history is None:
    st.session_state.history = load_history()
if st.session_state.profile is None:
    st.session_state.profile = load_profile()

@st.cache_resource
def get_model():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True, max_length=128
    )

if st.session_state.dark_mode:
    bg        = "#1a0a2e"
    card      = "#2a1a4e"
    text      = "#eaeaea"
    subtext   = "#bbaacc"
    border    = "#3a2a5e"
    btn_bg    = "#7c4dff"
    btn_text  = "#ffffff"
    back_bg   = "#3a2a5e"
    back_text = "#eaeaea"
    opts_btn  = "#3a2a5e"
    input_bg  = "#2a1a4e"
    result_bg_map = {
        'positive': '#0a2a1a', 'neutral': '#1a0a2e', 'negative': '#2a0a0a'
    }
else:
    bg        = "#f0ebff"
    card      = "#ffffff"
    text      = "#1a0a2e"
    subtext   = "#6a5a8a"
    border    = "#d8cef0"
    btn_bg    = "#7c4dff"
    btn_text  = "#ffffff"
    back_bg   = "#ede5ff"
    back_text = "#1a0a2e"
    opts_btn  = "#ede5ff"
    input_bg  = "#ffffff"
    result_bg_map = {
        'positive': '#e8f5e9', 'neutral': '#ede5ff', 'negative': '#ffebee'
    }

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, .stApp {{
    font-family: 'Poppins', sans-serif !important;
    background-color: {bg} !important;
    color: {text} !important;
}}

.stApp {{
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cg stroke='%237c4dff' stroke-width='1.5' fill='none' opacity='0.08'%3E%3Ccircle cx='40' cy='40' r='20'/%3E%3Ccircle cx='200' cy='80' r='14'/%3E%3Ccircle cx='360' cy='40' r='18'/%3E%3Cpath d='M10 100 Q30 80 50 100 Q70 120 90 100'/%3E%3Cpath d='M150 150 Q170 130 190 150 Q210 170 230 150'/%3E%3Cpath d='M300 200 Q320 180 340 200 Q360 220 380 200'/%3E%3Crect x='60' y='160' width='30' height='30' rx='4' transform='rotate(15 75 175)'/%3E%3Crect x='250' y='60' width='24' height='24' rx='4' transform='rotate(-10 262 72)'/%3E%3Cpath d='M100 250 L120 230 L140 250 L120 270 Z'/%3E%3Cpath d='M320 300 L340 280 L360 300 L340 320 Z'/%3E%3Cpath d='M30 300 Q50 280 70 300'/%3E%3Cpath d='M180 350 Q200 330 220 350'/%3E%3Ccircle cx='160' cy='280' r='10'/%3E%3Ccircle cx='80' cy='360' r='16'/%3E%3Ccircle cx='300' cy='150' r='12'/%3E%3Cpath d='M200 200 Q220 180 240 200 Q260 220 280 200'/%3E%3C/g%3E%3C/svg%3E") !important;
    background-size: 400px 400px !important;
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 1rem !important;
    max-width: 1200px;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.app-title {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {text};
    text-align: center;
    margin: 6px 0 4px 0;
}}

.app-subtitle {{
    font-size: 0.95rem;
    color: {subtext};
    text-align: center;
    margin-bottom: 20px;
}}

.welcome-bar {{
    background: {card};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 10px 20px;
    margin-bottom: 16px;
    font-size: 0.95rem;
    color: {subtext};
    box-shadow: 0 2px 8px rgba(124,77,255,0.06);
}}

.card {{
    background: {card};
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid {border};
    box-shadow: 0 2px 16px rgba(124,77,255,0.08);
    margin-bottom: 14px;
}}

.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {text};
    margin-bottom: 12px;
}}

.sentiment-row {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 9px 13px;
    border-radius: 9px;
    margin: 5px 0;
    font-size: 0.9rem;
    font-weight: 500;
}}

.positive {{ background:#d4edda; color:#155724; }}
.negative {{ background:#f8d7da; color:#721c24; }}
.neutral  {{ background:#e8e0ff; color:#3a0a6e; }}

.shift-card {{
    padding: 10px 14px;
    border-left: 4px solid {btn_bg};
    border-radius: 8px;
    background: {opts_btn};
    margin: 7px 0;
    color: {text};
    font-size: 0.88rem;
}}

.overall-badge {{
    display: inline-block;
    padding: 7px 22px;
    border-radius: 30px;
    font-size: 1rem;
    font-weight: 700;
    margin: 6px auto;
    text-align: center;
}}

.history-item {{
    background: {opts_btn};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 11px 15px;
    margin: 7px 0;
    color: {text};
    font-size: 0.88rem;
}}

.summary-box {{
    background: {card};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 28px 32px;
    color: {text};
    font-size: 1rem;
    line-height: 1.9;
    box-shadow: 0 2px 16px rgba(124,77,255,0.08);
    margin-top: 10px;
}}

.error-box {{
    background: #fff0f0;
    border: 1.5px solid #e74c3c;
    border-radius: 10px;
    padding: 12px 18px;
    color: #c0392b;
    font-size: 0.92rem;
    margin-top: 10px;
}}

div.stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 9px 18px !important;
    font-family: 'Poppins', sans-serif !important;
    background-color: {btn_bg} !important;
    color: {btn_text} !important;
}}

div.stButton > button:hover,
div.stButton > button:focus {{
    background-color: {'#5a2abf' if st.session_state.dark_mode else '#5c35cc'} !important;
    color: #ffffff !important;
    opacity: 1 !important;
    border: none !important;
}}

.back-btn div.stButton > button {{
    background-color: {back_bg} !important;
    color: {back_text} !important;
    border: 1px solid {border} !important;
    font-size: 0.9rem !important;
    padding: 7px 14px !important;
}}

.back-btn div.stButton > button:hover {{
    background-color: {btn_bg} !important;
    color: #ffffff !important;
}}

.opts-panel div.stButton > button {{
    background-color: {opts_btn} !important;
    color: {text} !important;
    border: 1px solid {border} !important;
    width: 100% !important;
    margin-bottom: 6px !important;
}}

.opts-panel div.stButton > button:hover {{
    background-color: {btn_bg} !important;
    color: #ffffff !important;
}}

div.stTextArea textarea {{
    border-radius: 12px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.93rem !important;
    background: {input_bg} !important;
    color: {text} !important;
    border: 1.5px solid {border} !important;
}}

div.stTextArea label, div.stTextInput label {{
    color: {text} !important;
    font-weight: 600 !important;
}}

div.stTextInput input {{
    border-radius: 10px !important;
    background: {input_bg} !important;
    color: {text} !important;
    border: 1.5px solid {border} !important;
    font-family: 'Poppins', sans-serif !important;
}}
</style>
""", unsafe_allow_html=True)

# TOP BAR
col_back, col_space, col_dots = st.columns([2, 7, 1])

with col_back:
    if st.session_state.page != 'home':
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Go Back"):
            if st.session_state.page == 'summary':
                st.session_state.page = 'results'
            else:
                st.session_state.page = 'home'
            st.session_state.show_options = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with col_dots:
    if st.button("⋮", key="dots_btn"):
        st.session_state.show_options = not st.session_state.show_options
        st.rerun()

if st.session_state.show_options:
    _, opt_col, _ = st.columns([6, 3, 1])
    with opt_col:
        st.markdown('<div class="opts-panel card">', unsafe_allow_html=True)
        st.markdown(f'<p style="font-weight:700;color:{text};margin:0 0 10px 0;">⚙️ Options</p>', unsafe_allow_html=True)
        dark_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(dark_label, key="toggle_dark", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.session_state.show_options = False
            st.rerun()
        if st.button("👤 My Profile", key="go_profile", use_container_width=True):
            st.session_state.page = 'profile'
            st.session_state.show_options = False
            st.rerun()
        if st.button("🕒 Previous Analyses", key="go_history", use_container_width=True):
            st.session_state.page = 'history'
            st.session_state.show_options = False
            st.rerun()
        if st.button("✖ Close", key="close_opts", use_container_width=True):
            st.session_state.show_options = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ============================
# PAGE: HOME
# ============================
if st.session_state.page == 'home':

    profile = st.session_state.profile
    st.markdown(f"""
    <div class="welcome-bar">
        👋 Welcome back, <b style="color:{btn_bg};">{profile['name']}</b>!
        &nbsp;|&nbsp; {profile['college']} · {profile['branch']} · {profile['semester']}
        &nbsp;|&nbsp; 📊 {len(st.session_state.history)} conversation{'s' if len(st.session_state.history) != 1 else ''} analysed
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="app-title">🗣️ Hinglish Sentiment Shift Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Paste a Hinglish conversation and discover its emotional journey</div>', unsafe_allow_html=True)

    sample = st.session_state.conversation_text or """Rohan: Yaar aaj mera din bahut bura tha, sab kuch galat ho gaya.
Priya: Arre kya hua? Bata mujhe, I'm here for you.
Rohan: Office mein boss ne sabke saamne scold kiya, bahut bura laga.
Priya: Oh no! That's so unfair. Tu itna hardworking hai, boss ko sharam aani chahiye.
Rohan: Haan yaar, par ab thoda better feel ho raha hai tere se baat karke.
Priya: That's what friends are for! Aaj shaam ko chai peene chalte hai?
Rohan: Haan bilkul! Sounds great, definitely cheers me up."""

    conversation = st.text_area(
        "Enter Hinglish Conversation (Format — Speaker: message):",
        value=sample, height=260
    )
    st.session_state.conversation_text = conversation

    st.markdown(f'<p style="font-size:0.82rem;color:{subtext};margin-top:-8px;">💡 Each line should follow the format: <b>SpeakerName: message</b></p>', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        analyze = st.button("🔍 Analyze Sentiment Shifts", use_container_width=True)

    if analyze:
        if not conversation.strip():
            st.markdown('<div class="error-box">⚠️ Please enter a conversation first.</div>', unsafe_allow_html=True)
        else:
            turns = []
            bad_lines = []
            for i, line in enumerate(conversation.strip().split('\n')):
                line = line.strip()
                if not line:
                    continue
                if ':' not in line:
                    bad_lines.append(f"Line {i+1}: '{line}'")
                    continue
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text_content = parts[1].strip()
                if text_content:
                    turns.append({
                        'conv_id': 0, 'turn': i,
                        'speaker': speaker,
                        'text': text_content,
                        'clean_text': text_content.lower()
                    })

            if bad_lines:
                st.markdown(f'<div class="error-box">⚠️ These lines were skipped (missing "Speaker:" format):<br>{"<br>".join(bad_lines)}</div>', unsafe_allow_html=True)

            if len(turns) < 2:
                st.markdown('<div class="error-box">⚠️ Please enter at least 2 valid turns in format "Speaker: message"</div>', unsafe_allow_html=True)
            else:
                model = get_model()
                _, mid2, _ = st.columns([1, 2, 1])
                with mid2:
                    with st.spinner("⏳ Analyzing sentiments... please wait"):
                        df = pd.DataFrame(turns)
                        results = model(df['clean_text'].tolist())
                        df['sentiment'] = [r['label'].lower() for r in results]
                        df['sentiment_score'] = [round(r['score'], 3) for r in results]
                        shifts = detect_shifts(df)
                        overall = df['sentiment'].value_counts().index[0]

                        new_entry = {
                            'preview': conversation[:80] + '...',
                            'overall': overall,
                            'shifts': len(shifts),
                            'raw': conversation
                        }
                        st.session_state.history.append(new_entry)
                        save_history(st.session_state.history)

                        st.session_state.results = {
                            'df': df, 'shifts': shifts,
                            'overall': overall, 'raw': conversation
                        }
                        st.session_state.page = 'results'
                        st.rerun()

# ============================
# PAGE: RESULTS
# ============================
elif st.session_state.page == 'results':

    df      = st.session_state.results['df']
    shifts  = st.session_state.results['shifts']
    overall = st.session_state.results['overall']

    result_bg = result_bg_map.get(overall, bg)
    badge_bg  = {'positive': '#d4edda', 'neutral': '#e8e0ff', 'negative': '#f8d7da'}.get(overall, '#e8e0ff')
    badge_txt = {'positive': '#155724', 'neutral': '#3a0a6e', 'negative': '#721c24'}.get(overall, '#3a0a6e')
    emoji     = {'positive': '😊', 'neutral': '😐', 'negative': '😔'}.get(overall, '😐')

    if not st.session_state.dark_mode:
        st.markdown(f"<style>.stApp{{background-color:{result_bg}!important;}}</style>", unsafe_allow_html=True)

    st.markdown('<div class="app-title">📊 Analysis Results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:18px;">
        <span class="overall-badge" style="background:{badge_bg};color:{badge_txt};">
            {emoji} Overall Mood: {overall.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="card"><div class="section-title">💬 Sentiment per Turn</div>', unsafe_allow_html=True)
        for _, row in df.iterrows():
            em = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}.get(row['sentiment'], '⚪')
            st.markdown(f"""
            <div class="sentiment-row {row['sentiment']}">
                {em} <b>{row['speaker']}:</b>&nbsp;{row['text']}&nbsp;<i>({row['sentiment']})</i>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><div class="section-title">🔄 Shifts Detected</div>', unsafe_allow_html=True)
        if len(shifts) == 0:
            st.info("No sentiment shifts detected.")
        else:
            st.success(f"✅ Found {len(shifts)} sentiment shifts!")
            for _, s in shifts.iterrows():
                st.markdown(f"""
                <div class="shift-card">
                    <b>{s['shift_type']}</b><br>
                    <span style="color:{subtext}"><i>{s['speaker']}</i>: "{s['curr_text']}"</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    speakers_in_conv = df['speaker'].unique().tolist()
    graph_title = f"Sentiment Journey — {' vs '.join(speakers_in_conv)}"

    st.markdown(f'<div class="card" style="padding-top:10px;"><p class="section-title" style="text-align:center;">📈 {graph_title}</p>', unsafe_allow_html=True)

    sentiment_map_num = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['sentiment_num'] = df['sentiment'].map(sentiment_map_num)
    color_map = {'positive': '#2ecc71', 'neutral': '#7c4dff', 'negative': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_bg = '#1a0a2e' if st.session_state.dark_mode else '#faf8ff'
    fig.patch.set_facecolor(plot_bg)
    ax.set_facecolor(plot_bg)

    for i in range(len(df) - 1):
        x1, y1 = df.iloc[i]['turn'], df.iloc[i]['sentiment_num']
        x2, y2 = df.iloc[i+1]['turn'], df.iloc[i+1]['sentiment_num']
        seg_color = color_map.get(df.iloc[i]['sentiment'], '#7c4dff')
        ax.plot([x1, x2], [y1, y2], color=seg_color, linewidth=4, zorder=3, solid_capstyle='round')
        ax.fill_between([x1, x2], [y1, y2], -1.5, alpha=0.12, color=seg_color)

    for _, row in df.iterrows():
        c = color_map.get(row['sentiment'], '#7c4dff')
        ax.scatter(row['turn'], row['sentiment_num'],
                  color=c, s=220, zorder=6,
                  edgecolors='white', linewidths=3)
        ax.annotate(row['speaker'],
                   (row['turn'], row['sentiment_num']),
                   textcoords="offset points",
                   xytext=(0, 16), ha='center',
                   fontsize=9, fontweight='bold',
                   color=text if not st.session_state.dark_mode else '#eaeaea')

    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['😔 Negative', '😐 Neutral', '😊 Positive'],
                      fontsize=12, fontweight='bold',
                      color=text if not st.session_state.dark_mode else '#eaeaea')
    ax.set_xlabel('Conversation Turn',
                 color=text if not st.session_state.dark_mode else '#eaeaea', fontsize=11)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(colors=text if not st.session_state.dark_mode else '#eaeaea')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.12, linestyle='--', color='gray')

    legend = [
        mpatches.Patch(color='#2ecc71', label='Positive'),
        mpatches.Patch(color='#7c4dff', label='Neutral'),
        mpatches.Patch(color='#e74c3c', label='Negative')
    ]
    ax.legend(handles=legend, loc='upper right', framealpha=0.2,
             labelcolor=text if not st.session_state.dark_mode else '#eaeaea', fontsize=10)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    _, sum_col, _ = st.columns([1, 2, 1])
    with sum_col:
        if st.button("📝 View Summary", use_container_width=True):
            st.session_state.page = 'summary'
            st.rerun()

# ============================
# PAGE: SUMMARY
# ============================
elif st.session_state.page == 'summary':

    df      = st.session_state.results['df']
    shifts  = st.session_state.results['shifts']
    overall = st.session_state.results['overall']
    raw     = st.session_state.results.get('raw', '')

    result_bg = result_bg_map.get(overall, bg)
    badge_bg  = {'positive': '#d4edda', 'neutral': '#e8e0ff', 'negative': '#f8d7da'}.get(overall, '#e8e0ff')
    badge_txt = {'positive': '#155724', 'neutral': '#3a0a6e', 'negative': '#721c24'}.get(overall, '#3a0a6e')
    emoji     = {'positive': '😊', 'neutral': '😐', 'negative': '😔'}.get(overall, '😐')

    if not st.session_state.dark_mode:
        st.markdown(f"<style>.stApp{{background-color:{result_bg}!important;}}</style>", unsafe_allow_html=True)

    st.markdown('<div class="app-title">📝 Conversation Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:20px;">
        <span class="overall-badge" style="background:{badge_bg};color:{badge_txt};">
            {emoji} Overall Mood: {overall.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)

    lines = []
    for line in raw.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            lines.append((parts[0].strip(), parts[1].strip()))

    speakers    = list(dict.fromkeys([l[0] for l in lines]))
    total_turns = len(lines)
    sentiments  = df['sentiment'].tolist()
    shift_count = len(shifts)

    spk1 = speakers[0] if len(speakers) > 0 else "Person 1"
    spk2 = speakers[1] if len(speakers) > 1 else "Person 2"

    start_sent = sentiments[0] if sentiments else 'neutral'
    end_sent   = sentiments[-1] if sentiments else 'neutral'
    pos_count  = sentiments.count('positive')
    neg_count  = sentiments.count('negative')

    if start_sent == 'negative':
        opening = f"{spk1} opens the conversation sharing something difficult or upsetting."
    elif start_sent == 'positive':
        opening = f"{spk1} starts the conversation on a cheerful and enthusiastic note."
    else:
        opening = f"{spk1} begins the conversation in a calm, matter-of-fact way."

    if neg_count > 0 and pos_count > 0:
        middle = f"As the conversation progresses, emotions fluctuate — there are moments of tension as well as warmth and encouragement. {spk2} actively engages and responds to what {spk1} shares."
    elif pos_count > total_turns * 0.6:
        middle = f"The conversation remains largely warm and engaging throughout, with {spk2} responding positively and keeping the energy upbeat."
    elif neg_count > total_turns * 0.4:
        middle = f"The conversation carries a heavier emotional tone, with both speakers navigating a difficult or sensitive topic."
    else:
        middle = f"The conversation flows steadily, with both speakers exchanging thoughts in a relaxed and balanced manner."

    if end_sent == 'positive':
        ending = f"By the end, the mood lifts and the conversation closes on a positive and hopeful note."
    elif end_sent == 'negative':
        ending = f"The conversation ends on a heavier note, leaving some emotional tension unresolved."
    else:
        ending = f"The conversation wraps up in a calm and balanced way."

    if shift_count > 3:
        shift_insight = f"Overall, the emotional journey is dynamic, with {shift_count} mood shifts reflecting a rich and evolving dialogue."
    elif shift_count > 0:
        shift_insight = f"The conversation sees {shift_count} notable emotional shift{'s' if shift_count > 1 else ''}, marking clear turning points in the dialogue."
    else:
        shift_insight = "The emotional tone remains consistent from start to finish."

    summary_text = f"{opening} {middle} {ending} {shift_insight}"

    _, card_col, _ = st.columns([0.5, 9, 0.5])
    with card_col:
        st.markdown(f"""
        <div class="summary-box">
            <p style="font-size:1.05rem;font-weight:700;color:{text};margin-bottom:16px;">
                🗣️ What happened in this conversation?
            </p>
            <p style="line-height:1.9;font-size:1rem;color:{text};">
                {summary_text}
            </p>
            <hr style="border:none;border-top:1px solid {border};margin:18px 0;">
            <p style="font-size:0.85rem;color:{subtext};">
                👥 Speakers: <b>{' & '.join(speakers)}</b> &nbsp;|&nbsp;
                💬 Total turns: <b>{total_turns}</b> &nbsp;|&nbsp;
                🔄 Mood shifts: <b>{shift_count}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================
# PAGE: HISTORY
# ============================
elif st.session_state.page == 'history':

    st.markdown('<div class="app-title">🕒 Previous Analyses</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your past conversation analyses — saved across sessions</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown(f'<div class="card" style="text-align:center;color:{subtext};">No previous analyses yet! Go analyze a conversation first.</div>', unsafe_allow_html=True)
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - i
            em  = {'positive': '😊', 'neutral': '😐', 'negative': '😔'}.get(item['overall'], '😐')
            st.markdown(f"""
            <div class="history-item">
                <b>#{idx}</b> &nbsp;{em} <b>{item['overall'].upper()}</b>
                &nbsp;|&nbsp; {item['shifts']} shifts detected<br>
                <span style="color:{subtext};font-size:0.83rem;">{item['preview']}</span>
            </div>""", unsafe_allow_html=True)

            if st.button(f"Re-analyze #{idx}", key=f"reanalyze_{i}"):
                raw = item['raw']
                turns = []
                for j, line in enumerate(raw.strip().split('\n')):
                    if ':' in line:
                        parts = line.split(':', 1)
                        speaker = parts[0].strip()
                        text_content = parts[1].strip()
                        if text_content:
                            turns.append({
                                'conv_id': 0, 'turn': j,
                                'speaker': speaker,
                                'text': text_content,
                                'clean_text': text_content.lower()
                            })
                if turns:
                    with st.spinner("Re-analyzing..."):
                        model = get_model()
                        df = pd.DataFrame(turns)
                        results = model(df['clean_text'].tolist())
                        df['sentiment'] = [r['label'].lower() for r in results]
                        df['sentiment_score'] = [round(r['score'], 3) for r in results]
                        shifts = detect_shifts(df)
                        overall = df['sentiment'].value_counts().index[0]
                        st.session_state.results = {
                            'df': df, 'shifts': shifts,
                            'overall': overall, 'raw': raw
                        }
                        st.session_state.page = 'results'
                        st.rerun()

# ============================
# PAGE: PROFILE
# ============================
elif st.session_state.page == 'profile':

    st.markdown('<div class="app-title">👤 My Profile</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">', unsafe_allow_html=True)

    profile = st.session_state.profile

    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 10px 0;">
        <div style="font-size:4rem;">👤</div>
        <div style="font-size:1.3rem;font-weight:700;color:{text};">{profile['name']}</div>
        <div style="color:{subtext};font-size:0.9rem;margin-top:4px;">{profile['college']} · {profile['branch']} · {profile['semester']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        name     = st.text_input("Your Name", value=profile['name'])
        college  = st.text_input("College", value=profile['college'])
    with c2:
        branch   = st.text_input("Branch", value=profile['branch'])
        semester = st.text_input("Semester", value=profile['semester'])

    _, save_col, _ = st.columns([1, 2, 1])
    with save_col:
        if st.button("💾 Save Profile", use_container_width=True):
            new_profile = {'name': name, 'college': college, 'branch': branch, 'semester': semester}
            st.session_state.profile = new_profile
            save_profile(name, college, branch, semester)
            st.success(f"✅ Profile saved! Welcome, {name}!")
            st.rerun()

    st.markdown("---")

    total        = len(st.session_state.history)
    total_shifts = sum(h['shifts'] for h in st.session_state.history)
    moods        = [h['overall'] for h in st.session_state.history]
    top_mood     = max(set(moods), key=moods.count) if moods else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Conversations Analyzed", total)
    c2.metric("Total Shifts Found", total_shifts)
    c3.metric("Most Common Mood", top_mood.upper() if top_mood != "N/A" else "N/A")
    st.markdown('</div>', unsafe_allow_html=True)