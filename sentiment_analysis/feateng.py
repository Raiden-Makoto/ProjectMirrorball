"""
NLP Feature Engineering for Lyrics Analysis.

This module extracts NLP features from song lyrics including:
- VADER sentiment scores (compound, positive, negative)
- Lexical complexity (unique words / total words)
- Word count

These features are then joined with legacy track data to create
a master training dataset for ML models.
"""

import os
from typing import Dict, List, Optional

import duckdb  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


def get_lexical_complexity(text: Optional[str]) -> float:
    """
    Calculate lexical complexity (vocabulary diversity) of text.

    Lexical complexity is defined as the ratio of unique words to total words.
    Higher values indicate more diverse vocabulary.

    Args:
        text: Input text to analyze. Can be None or empty.

    Returns:
        Lexical complexity score between 0 and 1. Returns 0 if text is empty.
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if len(words) == 0:
        return 0.0

    return len(set(words)) / len(words)


def process_features() -> None:
    """
    Extract NLP features from lyrics and create training dataset.

    This function:
    1. Fetches all lyrics from dim_lyrics table
    2. Extracts VADER sentiment scores and text metrics for each track
    3. Saves features to dim_nlp_features table
    4. Creates master_training_data by joining NLP features with legacy track data

    The master_training_data table combines:
    - Legacy track metadata (track_name, album_name)
    - Audio features (valence, energy) from seed CSV
    - NLP features (sentiment, complexity, word_count) from lyrics analysis
    """
    conn = duckdb.connect(DB_PATH)

    # 1. Fetch lyrics from database
    print("Fetching lyrics from database...")
    df = conn.execute("SELECT track_name, album_name, lyrics FROM dim_lyrics").df()

    features: List[Dict[str, float]] = []

    # 2. Extract NLP features for each track
    print("Extracting NLP features...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lyrics = row["lyrics"]

        if lyrics:
            # VADER Sentiment Analysis
            vs = analyzer.polarity_scores(lyrics)

            # Text Metrics
            complexity = get_lexical_complexity(lyrics)
            word_count = len(lyrics.split())

            features.append(
                {
                    "track_name": row["track_name"],
                    "album_name": row["album_name"],
                    "sentiment_compound": vs["compound"],
                    "sentiment_pos": vs["pos"],
                    "sentiment_neg": vs["neg"],
                    "lexical_complexity": complexity,
                    "word_count": word_count,
                }
            )

    # 3. Save features to database
    print("\nSaving NLP features to database...")
    features_df = pd.DataFrame(features)
    conn.execute("CREATE OR REPLACE TABLE dim_nlp_features AS SELECT * FROM features_df")

    # 4. Create the "Master View" for the ML Model
    # This joins Legacy CSV stats with the new NLP features
    print("Creating master training dataset...")
    conn.execute(
        """
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
        """
    )

    # Report completion
    sample_count = conn.execute(
        "SELECT COUNT(*) FROM master_training_data"
    ).fetchone()[0]

    print("\n--- FEATURE ENGINEERING COMPLETE ---")
    print(f"Master Training Set created with {sample_count} samples.")
    conn.close()


if __name__ == "__main__":
    process_features()
