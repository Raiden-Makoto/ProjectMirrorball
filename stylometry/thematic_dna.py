"""
Thematic DNA analysis for Project Mirrorball

This model uses TF-IDF to identify the "signature words" that distinguish
the different "Eras" of Taylor Swift's songwriting
"""

import os
import duckdb # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from tqdm import tqdm # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def thematic_dna_analysis() -> None:
    """
    Main function to run the thematic DNA analysis.

    This function:
    1. Loads all lyrics from dim_lyrics table
    2. Initializes the TF-IDF vectorizer with stop words and section tag filtering
    3. Extracts the top 5 most characteristic words for each track (highest TF-IDF scores)
    4. Saves the results to dim_thematic_dna table
    5. Updates the master_training_data table with the thematic DNA

    TF-IDF identifies words that are distinctive to each track by measuring:
    - High term frequency (word appears often in this track)
    - Low document frequency (word appears rarely across all tracks)
    This captures the "signature words" that define each song's thematic identity.
    """
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT track_name, album_name, lyrics FROM dim_lyrics").df()
    
    # 1. Initialize Vectorizer (Filtering out 'Stop Words' like 'the', 'is', 'yeah')
    # We also filter out 'Taylor' and 'Swift' as they are metadata
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + ['taylor', 'swift', 'chorus', 'verse', 'bridge']
    
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=2000)
    matrix = tfidf.fit_transform(df['lyrics'].fillna(""))
    features = tfidf.get_feature_names_out()
    
    dna_results = []
    print("Extracting thematic DNA (TF-IDF signature words)...")
    for i in tqdm(range(len(df)), total=len(df)):
        row = matrix.getrow(i).toarray()[0]
        # Get the top 5 most characteristic words for this specific song (highest TF-IDF scores)
        top_indices = row.argsort()[-5:][::-1]
        keywords = [features[idx] for idx in top_indices if row[idx] > 0]
        
        dna_results.append({
            'track_name': df.iloc[i]['track_name'],
            'album_name': df.iloc[i]['album_name'],
            'thematic_dna': ", ".join(keywords)
        })

    # Save to Database
    dna_df = pd.DataFrame(dna_results)
    conn.execute("CREATE OR REPLACE TABLE dim_thematic_dna AS SELECT * FROM dna_df")
    
    # Update Master Table
    master_cols = [c[1] for c in conn.execute("PRAGMA table_info('master_training_data')").fetchall()]
    if 'thematic_dna' not in master_cols:
        conn.execute("ALTER TABLE master_training_data ADD COLUMN thematic_dna TEXT")
    
    conn.execute("""
        UPDATE master_training_data 
        SET thematic_dna = (
            SELECT s.thematic_dna FROM dim_thematic_dna s 
            WHERE s.track_name = master_training_data.track_name 
            AND s.album_name = master_training_data.album_name
        )
    """)
    
    print("\nThematic DNA extracted. Your master dataset now has identity.")
    conn.close()

if __name__ == "__main__":
    thematic_dna_analysis()