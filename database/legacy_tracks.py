import pandas as pd # type: ignore
import duckdb # type: ignore
import os

# 1. Initialize our 'Serverless' Database
DB_PATH = os.path.join(os.getcwd(), "mirrorball.db")
SEED_PATH = os.path.join(os.getcwd(), "data", "mirrorball_seed.csv")
conn = duckdb.connect(DB_PATH)

# 2. Load the Seed Data into DuckDB
# This converts our CSV into a high-performance relational table
df_seed = pd.read_csv(SEED_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS legacy_tracks AS SELECT * FROM df_seed")

print(f"--- DATABASE INITIALIZED: {len(df_seed)} tracks loaded ---")

# 3. The Resolver logic for tomorrow's mapping
def get_legacy_stats(track_name):
    """
    Looks up the Valence/Energy for an OG track to use as training labels.
    """
    # Simple logic to strip (Taylor's Version) for matching
    clean_name = track_name.replace("(Taylor's Version)", "").strip()
    
    query = f"SELECT valence, energy FROM legacy_tracks WHERE track_name = ? LIMIT 1"
    result = conn.execute(query, [clean_name]).fetchone()
    return result

# Quick test
sample_tv = "Love Story (Taylor's Version)"
stats = get_legacy_stats(sample_tv)
print(f"Mapping '{sample_tv}' to legacy stats: {stats}")