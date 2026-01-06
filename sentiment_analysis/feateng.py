import os
import duckdb # type: ignore
import pandas as pd # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
from tqdm import tqdm # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

# Initialize Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_lexical_complexity(text):
    if not text: return 0
    words = text.lower().split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)

def process_features():
    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch lyrics
    print("Fetching lyrics from database...")
    df = conn.execute("SELECT track_name, album_name, lyrics FROM dim_lyrics").df()
    
    features = []
    
    print("Extracting NLP features...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lyrics = row['lyrics']
        
        if lyrics:
            # VADER Sentiment
            vs = analyzer.polarity_scores(lyrics)
            
            # Text Metrics
            complexity = get_lexical_complexity(lyrics)
            word_count = len(lyrics.split())
            
            features.append({
                'track_name': row['track_name'],
                'album_name': row['album_name'],
                'sentiment_compound': vs['compound'],
                'sentiment_pos': vs['pos'],
                'sentiment_neg': vs['neg'],
                'lexical_complexity': complexity,
                'word_count': word_count
            })
    
    # 2. Save to a new table
    features_df = pd.DataFrame(features)
    conn.execute("CREATE OR REPLACE TABLE dim_nlp_features AS SELECT * FROM features_df")
    
    # 3. Create the "Master View" for the ML Model
    # This joins your Legacy CSV stats with the new NLP features
    conn.execute("""
        CREATE OR REPLACE TABLE master_training_data AS
        SELECT 
            l.track_name, 
            l.album_name,
            l.valence, 
            l.energy, 
            n.sentiment_compound, 
            n.lexical_complexity, 
            n.word_count,
            n.sentiment_pos,
            n.sentiment_neg
        FROM legacy_tracks l
        JOIN dim_nlp_features n 
          ON LOWER(l.track_name) = LOWER(n.track_name)
    """)
    
    print("\n--- FEATURE ENGINEERING COMPLETE ---")
    print(f"Master Training Set created with {conn.execute('SELECT COUNT(*) FROM master_training_data').fetchone()[0]} samples.")
    conn.close()

if __name__ == "__main__":
    process_features()