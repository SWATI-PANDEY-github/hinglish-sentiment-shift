from transformers import pipeline
import pandas as pd

def load_sentiment_model():
    print("Loading sentiment model...")
    sentiment = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=128
    )
    print("Model loaded!")
    return sentiment

def label_sentiments(df, sentiment_model):
    texts = df['clean_text'].tolist()
    print(f"Labeling {len(texts)} turns...")
    results = sentiment_model(texts, batch_size=64)
    df['sentiment'] = [r['label'].lower() for r in results]
    df['sentiment_score'] = [round(r['score'], 3) for r in results]
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/conversations.csv")
    
    # Take only 200 conversations instead of all 1589
    top_convs = df['conv_id'].unique()[:200]
    df = df[df['conv_id'].isin(top_convs)]
    print(f"Using {len(df)} turns from 200 conversations")
    
    model = load_sentiment_model()
    df = label_sentiments(df, model)
    print(df[['speaker', 'text', 'sentiment']].head(10).to_string())
    df.to_csv("data/processed/conversations_labeled.csv", index=False)
    print("Saved to data/processed/conversations_labeled.csv")