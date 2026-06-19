import os
import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def parse_conversation(filepath):
    turns = []
    conv_id = os.path.basename(filepath).replace('.txt', '')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip()
                if text:
                    turns.append({
                        'conv_id': conv_id,
                        'turn': i,
                        'speaker': speaker,
                        'text': text,
                        'clean_text': clean_text(text)
                    })
    except Exception as e:
        pass
    return turns

def load_all_conversations(folder):
    all_turns = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            filepath = os.path.join(folder, file)
            turns = parse_conversation(filepath)
            all_turns.extend(turns)
    return pd.DataFrame(all_turns)

if __name__ == "__main__":
    df = load_all_conversations("data/raw/hinglish-conv/conversations")
    print(f"Total turns loaded: {len(df)}")
    print(df.head(10))
    df.to_csv("data/processed/conversations.csv", index=False)
    print("Saved to data/processed/conversations.csv")