"""
Legacy Tracks Database Initialization.

This module loads the seed CSV data into DuckDB and provides
a lookup function for retrieving valence/energy scores for legacy tracks.

The legacy_tracks table serves as the ground truth data for training
ML models to predict valence/energy for new era tracks.
"""

import os
from typing import Optional, Tuple

import duckdb  # type: ignore
import pandas as pd  # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")
SEED_PATH = os.path.join(PROJECT_ROOT, "data", "mirrorball_seed.csv")

# 1. Initialize our 'Serverless' Database
conn = duckdb.connect(DB_PATH)

# 2. Load the Seed Data into DuckDB
# This converts our CSV into a high-performance relational table
df_seed = pd.read_csv(SEED_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS legacy_tracks AS SELECT * FROM df_seed")

print(f"--- DATABASE INITIALIZED: {len(df_seed)} tracks loaded ---")


# 3. The Resolver logic for mapping tracks to legacy stats
def get_legacy_stats(track_name: str) -> Optional[Tuple[float, float]]:
    """
    Looks up the Valence/Energy for an original track to use as training labels.

    This function handles the mapping between "Taylor's Version" tracks and
    their original versions by stripping the "(Taylor's Version)" suffix.

    Args:
        track_name: Name of the track (can include "(Taylor's Version)")

    Returns:
        Tuple of (valence, energy) if found, None otherwise
    """
    # Simple logic to strip (Taylor's Version) for matching
    clean_name = track_name.replace("(Taylor's Version)", "").strip()

    query = "SELECT valence, energy FROM legacy_tracks WHERE track_name = ? LIMIT 1"
    result = conn.execute(query, [clean_name]).fetchone()
    return result


# Quick test
if __name__ == "__main__":
    sample_tv = "Love Story (Taylor's Version)"
    stats = get_legacy_stats(sample_tv)
    print(f"Mapping '{sample_tv}' to legacy stats: {stats}")
