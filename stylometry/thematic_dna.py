"""
Thematic DNA analysis for Project Mirrorball

This model uses TF-IDF to identify the "signature words" that distinguish
the different "Eras" of Taylor Swift's songwriting
"""

import os
import re
import duckdb # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from tqdm import tqdm # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def deep_clean(text: str) -> str:
    """
    Pre-process lyrics to remove metadata noise before vectorization.
    
    Removes:
    - Anything in brackets [Artist Name] or [Verse: Artist]
    - Specific artist names if they appear as standalone lines (Genius artifact)
    """
    if not text:
        return ""
    # Remove anything in brackets [Artist Name] or [Verse: Artist]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove the specific artist names if they appear as standalone lines (Genius artifact)
    text = re.sub(r'(?i)\b(post malone|taylor swift|jack antonoff|florence\s*\+\s*the\s*machine)\b', '', text)
    return text

def thematic_dna_analysis() -> None:
    """
    Main function to run the thematic DNA analysis.

    This function:
    1. Loads all lyrics from dim_lyrics table
    2. Applies deep cleaning layer:
       - Removes bracket content [Artist Name] or [Verse: Artist] via regex
       - Removes specific artist names (Post Malone, Taylor Swift, Jack Antonoff, etc.)
    3. Applies data sanitization layer:
       - Dynamically extracts metadata words from track and album names (e.g., 'Malone', 'Taylor', 'Version')
       - Explicitly blacklists specific words ('post', 'malone', 'florence', etc.)
       - Filters out common Genius noise words and fragments ('ve', 'll', 're', 'nt', 'lyrics', 'embed', etc.)
       - Combines with standard English stop words
    4. Initializes the TF-IDF vectorizer with refined filtering:
       - Token pattern requiring minimum 3 letters (no numbers, no fragments)
       - Single-word ngrams only (ngram_range=(1, 1))
       - Maximum 2,000 features
    5. Extracts the top 5 most characteristic words for each track (highest TF-IDF scores)
    6. Saves the results to dim_thematic_dna table
    7. Updates the master_training_data table with the thematic DNA

    TF-IDF identifies words that are distinctive to each track by measuring:
    - High term frequency (word appears often in this track)
    - Low document frequency (word appears rarely across all tracks)
    This captures the "signature words" that define each song's thematic identity.
    """
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT track_name, album_name, lyrics FROM dim_lyrics").df()
    
    # --- THE CLEANING LAYER: Pre-process lyrics to remove metadata noise ---
    print("Cleaning lyrics (removing brackets and artist names)...")
    df['cleaned_lyrics'] = df['lyrics'].apply(deep_clean)
    
    # --- DATA SANITIZATION LAYER ---
    # 1. Get all words from Track and Album names to exclude (e.g., 'Malone', 'Taylor', 'Version')
    metadata_words = set()
    for col in ['track_name', 'album_name']:
        for item in df[col].dropna():
            words = re.findall(r'\b[a-zA-Z]{3,}\b', item.lower())
            metadata_words.update(words)
            
    # 2. THE BLACKLIST: Explicitly force-drop specific words
    custom_stops = [
        'post', 'malone', 'florence', 'machine', 'antonoff', 'dessner',
        'taylor', 'swift', 'version', 'lyrics', 'embed', 'contributors',
        'translations', 'translated', 'copy', 'link', 've', 'll', 're', 'nt',
        'oh', 'yeah', 'ah', 'ooh', 'chorus', 'verse', 'bridge'
    ]
    
    # 3. Combine into a master stop-word list
    all_custom_stops = list(metadata_words) + custom_stops
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + all_custom_stops
    # -------------------------------
    
    # 4. Refined Vectorizer with token pattern to ensure we only get words (minimum 3 letters, no numbers)
    # This filters out fragments and ensures meaningful words only
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',  # Minimum 3 letters, no numbers
        max_features=2000,
        ngram_range=(1, 1)  # Stick to single words for DNA
    )
    matrix = tfidf.fit_transform(df['cleaned_lyrics'].fillna(""))
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
    
    print("--- 🧬 DNA SANITIZED: Metadata noise removed ---")
    
    # Verify 'Fortnight'
    check = conn.execute("SELECT thematic_dna FROM dim_thematic_dna WHERE track_name LIKE 'Fortnight%'").fetchone()
    print(f"Fortnight DNA (Verified): {check[0] if check else 'Not Found'}")
    
    conn.close()

if __name__ == "__main__":
    thematic_dna_analysis()