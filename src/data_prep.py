import pandas as pd
import numpy as np
import os
import glob
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

def calculate_volatility(df):
    """Bitcoin ki daily returns aur volatility calculate karta hai"""
    # Sab columns ko lowercase karein taake case-sensitivity ka error na aaye
    df.columns = df.columns.astype(str).str.strip().str.lower()
    
    if 'date' in df.columns: df.rename(columns={'date': 'Date'}, inplace=True)
    if 'close' in df.columns: df.rename(columns={'close': 'Close'}, inplace=True)
    if 'price' in df.columns and 'Close' not in df.columns: df.rename(columns={'price': 'Close'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    
    if df['Close'].dtype == 'object' or df['Close'].dtype == 'string':
        df['Close'] = df['Close'].astype(str).str.replace(',', '').str.replace('$', '').str.strip()
    
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_7d'] = df['Daily_Return'].rolling(window=7).std()
    
    return df[['Date', 'Close', 'Daily_Return', 'Volatility_7d']].dropna()

def get_sentiment(text):
    """TextBlob use kar ke sentiment score nikalta hai"""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def process_text_dataframe(df, date_col_hint, text_col_hint):
    """Smart Column Detection ke sath text dataframe process karta hai"""
    df.columns = df.columns.astype(str).str.strip().str.lower()
    
    actual_date_col = 'date'
    actual_text_col = 'text'
    
    # Date column automatically dhoondna
    for col in [date_col_hint.lower(), 'date', 'datetime', 'time']:
        if col in df.columns:
            actual_date_col = col
            break
            
    # Text ya Title column automatically dhoondna
    for col in [text_col_hint.lower(), 'title', 'text', 'headline', 'content', 'tweet']:
        if col in df.columns:
            actual_text_col = col
            break
            
    # Agar column phir bhi na mile toh properly batao
    if actual_date_col not in df.columns or actual_text_col not in df.columns:
        print(f"\n[ERROR] Found these columns in the dataset: {df.columns.tolist()}")
        raise ValueError("Could not find exact date or text columns. Please check your CSV files.")

    df['Date'] = pd.to_datetime(df[actual_date_col], errors='coerce').dt.date
    df['Date'] = pd.to_datetime(df['Date']) 
    df = df.dropna(subset=['Date', actual_text_col])
    
    # Text ko string mein convert zaroor karein taake float errors na aayen
    df['Sentiment'] = df[actual_text_col].astype(str).apply(get_sentiment)
    
    daily_sentiment = df.groupby('Date')['Sentiment'].mean().reset_index()
    return daily_sentiment

def merge_datasets():
    print("1. Processing Historical Price Data...")
    history_file = os.path.join(RAW_DIR, 'history', 'bitcoin_history.csv') 
    df_history = pd.read_csv(history_file)
    df_history = calculate_volatility(df_history)
    
    print("2. Processing News Data (This will combine multiple files)...")
    news_files = glob.glob(os.path.join(RAW_DIR, 'news', '*.csv'))
    df_news_list = []
    
    for file in news_files:
        try:
            temp_df = pd.read_csv(file, on_bad_lines='skip', engine='python')
            df_news_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            
    df_news_raw = pd.concat(df_news_list, ignore_index=True)
    df_news = process_text_dataframe(df_news_raw, date_col_hint='date', text_col_hint='title')
    df_news.rename(columns={'Sentiment': 'News_Sentiment'}, inplace=True)
    print(f"News data processed. Total unique dates: {len(df_news)}")
    
    print("3. Processing Tweets Data (This might take a few minutes)...")
    tweets_file = os.path.join(RAW_DIR, 'tweets', 'tweets.csv')
    df_tweets_raw = pd.read_csv(tweets_file, lineterminator='\n', on_bad_lines='skip', low_memory=False)
    df_tweets = process_text_dataframe(df_tweets_raw, date_col_hint='date', text_col_hint='text')
    df_tweets.rename(columns={'Sentiment': 'Tweet_Sentiment'}, inplace=True)
    
    print("4. Merging all datasets into one...")
    merged_df = pd.merge(df_history, df_news, on='Date', how='left')
    merged_df = pd.merge(merged_df, df_tweets, on='Date', how='left')
    
    merged_df.fillna({'News_Sentiment': 0, 'Tweet_Sentiment': 0}, inplace=True)
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    output_path = os.path.join(PROCESSED_DIR, 'merged_data.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\n✅ Success! Final data saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()