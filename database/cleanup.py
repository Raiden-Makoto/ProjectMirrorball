"""
Database Cleanup Script

Drops intermediate tables and keeps only:
- Raw dimension tables (dim_*)
- Final output tables (final_*)
"""

import os
import duckdb # type: ignore

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def cleanup_database() -> None:
    """Remove intermediate tables, keep only dim_* and final_* tables."""
    conn = duckdb.connect(DB_PATH) # type: ignore
    
    # Tables to keep
    keep_tables = [
        'dim_bridge_metrics',
        'dim_lexical_metrics', 
        'dim_lyrics',
        'dim_nlp_features',
        'dim_thematic_dna',
        'final_analytical_set',
        'final_map_data_with_shap',
        'legacy_tracks',  # Keep for reference
        'master_training_data'  # Keep as it's a key table
    ]
    
    # Get all tables
    all_tables = conn.execute("SHOW TABLES").df()['name'].tolist()
    
    # Find tables to drop
    tables_to_drop = [t for t in all_tables if t not in keep_tables]
    
    if tables_to_drop:
        print(f"Dropping {len(tables_to_drop)} intermediate tables:")
        for table in tables_to_drop:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  ✓ Dropped {table}")
            except Exception as e:
                print(f"  ✗ Error dropping {table}: {e}")
    else:
        print("No intermediate tables to drop.")
    
    # Show remaining tables
    remaining = conn.execute("SHOW TABLES").df()
    print(f"\nRemaining tables ({len(remaining)}):")
    print(remaining.to_markdown(index=False))
    
    conn.close()
    print("\nDatabase cleanup complete.")

if __name__ == "__main__":
    cleanup_database()
