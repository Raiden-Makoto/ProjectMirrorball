import duckdb # type: ignore
import os

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def unify_data_v2() -> None:
    """
    Create a Master View of EVERYTHING we have lyrics for.
    Includes all 333 tracks - 231 with labels, 102 without (target for prediction).
    """
    conn = duckdb.connect(DB_PATH) # type: ignore
    
    # 1. Create a Master View of EVERYTHING we have lyrics for
    conn.execute("""
    CREATE OR REPLACE TABLE final_analytical_set AS
    SELECT 
        l.track_name, 
        l.album_name, 
        m.energy,         -- Will be NULL for the 102 tracks
        m.valence,        -- Will be NULL for the 102 tracks
        lx.reading_grade, 
        lx.syllable_density, 
        lx.lexical_diversity,
        COALESCE(b.bridge_sentiment_shift, 0) as bridge_shift
    FROM dim_lyrics l
    LEFT JOIN master_training_data m 
        ON l.track_name = m.track_name AND l.album_name = m.album_name
    LEFT JOIN dim_lexical_metrics lx 
        ON l.track_name = lx.track_name AND l.album_name = lx.album_name
    LEFT JOIN dim_bridge_metrics b 
        ON l.track_name = b.track_name AND l.album_name = b.album_name
    """)
    
    # Check coverage
    stats = conn.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE energy IS NOT NULL) as labeled,
            COUNT(*) FILTER (WHERE energy IS NULL) as unlabeled
        FROM final_analytical_set
    """).fetchone()
    
    print(f"Labeled (Training): {stats[0]} | Unlabeled (Target): {stats[1]}")
    conn.close()

if __name__ == "__main__":
    unify_data_v2()