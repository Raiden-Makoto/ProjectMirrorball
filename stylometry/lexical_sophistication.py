"""
Lexical Sophistication Analysis for Project Mirrorball

This module analyzes the lexical sophistication of Taylor Swift's songs.
We aim to mathematically map the stylistic shift from one genre to another.
Country and Pop (her early eras) rely on "Earworms" (short, punchy, monosyllabic words) that are easy to sing along to.
Folk and "Academic" Pop (her later eras) rely on "Storytelling"
(longer, multisyllabic words) that convey specific, often niche, imagery.

This analysis will help us understand how her lyrics have evolved stylistically over time.
It also serves as a feature engineering step for the master training dataset.
"""

import os
import duckdb  # type: ignore
import pandas as pd  # type: ignore
import textstat  # type: ignore
from tqdm import tqdm  # type: ignore

from typing import Dict, Optional

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def calculate_lexical_sophistication(lyrics: str) -> Optional[Dict[str, float]]:
    if not lyrics:
        return None
    # 1. Remove section tags for this analysis only
    clean_text = pd.Series([lyrics]).str.replace(r"\[.*?\]", "", regex=True)[0]
    
    # 2. THE FIX: Ensure every newline ends with a period
    # This forces textstat to recognize each line as a sentence.
    processed_text = ". ".join(clean_text.split("\n"))
    
    words = clean_text.split()
    
    # 3. Reading Grade (Now with realistic sentence counts)
    # Using processed_text ensures each line is treated as a sentence
    # Actual results show averages around 1-2 for most albums
    grade = textstat.flesch_kincaid_grade(processed_text)
    
    # calculate syllable density (words / syllables)
    syllables = textstat.syllable_count(clean_text)
    syllable_density = syllables / len(words) if len(words) > 0 else 0
    
    # calculate unique word count
    unique_words = len(set([w.lower().strip(".,!?") for w in words]))
    lexical_diversity = unique_words / len(words) if len(words) > 0 else 0
    
    # calculate "difficult words" percentage
    diff_words = textstat.difficult_words(clean_text)
    difficult_ratio = diff_words / len(words) if len(words) > 0 else 0
    
    return {
        "reading_grade": grade,
        "syllable_density": syllable_density,
        "lexical_diversity": lexical_diversity,
        "difficult_ratio": difficult_ratio,
    }

def main() -> None:
    """
    Main function to process all tracks and calculate lexical sophistication metrics.

    This function:
    1. Loads all lyrics from dim_lyrics table
    2. Calculates lexical sophistication metrics for each track
    3. Saves metrics to dim_lexical_sophistication table
    """
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT track_name, album_name, lyrics FROM dim_lyrics").df()
    
    lex_results = []
    print("Running Lexical Sophistication Analysis...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        metrics = calculate_lexical_sophistication(row['lyrics'])
        if metrics:
            metrics.update({
                'track_name': row['track_name'], 
                'album_name': row['album_name']
            })
            lex_results.append(metrics)
    
    # Save to database
    res_df = pd.DataFrame(lex_results)
    conn.execute("CREATE OR REPLACE TABLE dim_lexical_metrics AS SELECT * FROM res_df")
    
    # Update Master Table
    cols_to_add = ["reading_grade", "syllable_density", "lexical_diversity", "difficult_ratio"]
    master_cols = [c[1] for c in conn.execute("PRAGMA table_info('master_training_data')").fetchall()]
    
    for col in cols_to_add:
        if col not in master_cols:
            conn.execute(f"ALTER TABLE master_training_data ADD COLUMN {col} FLOAT")
        
        conn.execute(f"""
            UPDATE master_training_data 
            SET {col} = (
                SELECT s.{col} FROM dim_lexical_metrics s 
                WHERE s.track_name = master_training_data.track_name 
                AND s.album_name = master_training_data.album_name
            )
        """)
        
    print("\nLexical Metrics added to master_training_data.")
    
    # Insight Check: Highest Grade Level
    top_grade = conn.execute("""
        SELECT track_name, album_name, reading_grade 
        FROM dim_lexical_metrics 
        ORDER BY reading_grade DESC LIMIT 5
    """).fetchall()
    
    print("\nMost 'Sophisticated' Songs:")
    for track_name, album_name, grade in top_grade:
        print(f"  {track_name} ({album_name}) - Grade Level: {grade:.2f}")
    
    # Album-level analysis: Average reading grade per album
    print("\n" + "=" * 60)
    print("Average Reading Grade by Album:")
    print("=" * 60)
    album_stats = conn.execute("""
        SELECT album_name, 
               AVG(reading_grade) as avg_reading_grade,
               MIN(reading_grade) as min_reading_grade,
               MAX(reading_grade) as max_reading_grade,
               COUNT(*) as track_count
        FROM dim_lexical_metrics
        GROUP BY album_name
        ORDER BY avg_reading_grade DESC
    """).fetchall()
    
    for album_name, avg_grade, min_grade, max_grade, count in album_stats:
        print(f"{album_name:50s} | Avg: {avg_grade:5.2f} | Min: {min_grade:5.2f} | Max: {max_grade:5.2f} | Tracks: {count:2d}")
    
    conn.close()

if __name__ == "__main__":
    main()