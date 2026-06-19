import pandas as pd

def detect_shifts(df):
    """
    For each conversation, go turn by turn and detect
    when sentiment changes from one turn to the next.
    """
    all_shifts = []

    for conv_id, group in df.groupby('conv_id'):
        group = group.sort_values('turn').reset_index(drop=True)
        
        for i in range(1, len(group)):
            prev = group.iloc[i-1]
            curr = group.iloc[i]
            
            if prev['sentiment'] != curr['sentiment']:
                all_shifts.append({
                    'conv_id': conv_id,
                    'turn': curr['turn'],
                    'speaker': curr['speaker'],
                    'prev_sentiment': prev['sentiment'],
                    'curr_sentiment': curr['sentiment'],
                    'prev_text': prev['text'],
                    'curr_text': curr['text'],
                    'shift_type': f"{prev['sentiment']} → {curr['sentiment']}"
                })

    return pd.DataFrame(all_shifts)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/conversations_labeled.csv")
    
    shifts = detect_shifts(df)
    
    print(f"Total shifts detected: {len(shifts)}")
    print()
    print("Shift types:")
    print(shifts['shift_type'].value_counts())
    print()
    print("Sample shifts:")
    print(shifts[['speaker', 'prev_text', 'curr_text', 'shift_type']].head(5).to_string())
    
    shifts.to_csv("data/processed/shifts.csv", index=False)
    print()
    print("Saved to data/processed/shifts.csv")