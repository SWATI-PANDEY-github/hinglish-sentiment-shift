# 🗣️ Hinglish Sentiment Shift Detector

A research-level NLP project that detects **how sentiment shifts across turns in Hinglish (Hindi+English) conversations** — an underexplored problem in code-mixed language processing.

## 🔍 What is this?

Most sentiment analysis tools classify a single sentence as positive/negative/neutral. This project goes further — it tracks **how the emotional tone changes** across an entire conversation in Hinglish, the code-mixed language spoken by 500M+ Indians.

**Example:**
> Rohan: "Yaar aaj mera din bahut bura tha" → 😔 Negative  
> Priya: "Arre kya hua? I'm here for you" → 😊 Positive  
> **Shift detected: negative → positive**

## ✨ Features

- 🧠 Sentiment labeling using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- 🔄 Conversation-level shift detection across 6 shift types
- 📊 Interactive Streamlit app with colorful sentiment journey graph
- 📝 Auto-generated conversation summary
- 🕒 Analysis history saved across sessions
- 🌙 Dark/Light mode

## 📊 Dataset

- **Conversation data:** 1,589 Hinglish conversation files
- **Labeled turns:** 4,410 turns across 200 conversations
- **Shifts detected:** 1,098 sentiment shifts

| Shift Type | Count |
|---|---|
| neutral → positive | 544 |
| positive → neutral | 437 |
| neutral → negative | 44 |
| negative → neutral | 37 |
| negative → positive | 22 |
| positive → negative | 14 |

## 🛠️ Tech Stack

- Python, Pandas, Scikit-learn
- HuggingFace Transformers (RoBERTa)
- Streamlit
- Matplotlib

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/SWATI-PANDEY-github/hinglish-sentiment-shift.git
cd hinglish-sentiment-shift

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## 📁 Project Structure