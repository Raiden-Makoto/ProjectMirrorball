"""
XGBoost Model for Reconstructing Valence and Energy Scores.

This module implements the final data cleaning workflow:
1. Train XGBoost on 231 labeled tracks (with energy/valence)
2. Use reading_grade, lexical_diversity, bridge_shift as features
3. Predict energy and valence for 102 unlabeled tracks
4. Update final_analytical_set with AI-generated predictions
"""

import os
from typing import List

import duckdb  # type: ignore
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

# Feature columns used for training (stylometric features)
FEATURES: List[str] = [
    "reading_grade",
    "lexical_diversity",
    "bridge_shift",
]

# Target variables to predict
TARGETS: List[str] = ["valence", "energy"]


def train_and_predict() -> None:
    """
    Final Boss Workflow: Train on labeled tracks, predict for unlabeled tracks.
    
    This function:
    1. Loads final_analytical_set (all 333 tracks)
    2. Separates into labeled (231) and unlabeled (102) tracks
    3. Trains separate XGBoost models for valence and energy
    4. Validates models using MAE (Mean Absolute Error)
    5. Generates predictions for unlabeled tracks
    6. Updates final_analytical_set with AI-generated predictions
    """
    conn = duckdb.connect(DB_PATH)

    # 1. Load ALL data from final_analytical_set
    print("Loading final_analytical_set...")
    df_all = conn.execute("SELECT * FROM final_analytical_set").df()

    # 2. Split into labeled (training) and unlabeled (prediction)
    df_train = df_all[df_all['energy'].notna()].copy()
    df_predict = df_all[df_all['energy'].isna()].copy()

    print(f"Training on {len(df_train)} labeled tracks")
    print(f"Predicting for {len(df_predict)} unlabeled tracks")

    # Initialize results dataframe with track metadata for predictions
    results_df = df_predict[["track_name", "album_name"]].copy()

    # Train a separate model for each target variable
    for target in TARGETS:
        print(f"\n--- Training Model for {target.upper()} ---")

        X_train_full = df_train[FEATURES]
        y_train_full = df_train[target]

        # Split for validation (80/20 train/test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )

        # XGBoost Regressor optimized for Apple Silicon
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            tree_method="hist",  # Efficient on CPU
            device="cpu",
        )

        model.fit(X_train, y_train)

        # Validate model performance
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model Accuracy (MAE): {round(mae, 4)}")

        # Predict on unlabeled tracks
        X_predict = df_predict[FEATURES]
        results_df[target] = model.predict(X_predict)

    # 3. Update final_analytical_set with predictions
    print("\nUpdating final_analytical_set with predictions...")
    # Merge predictions back into the full dataset
    df_all_updated = df_all.copy()
    for _, row in results_df.iterrows():
        mask = (df_all_updated['track_name'] == row['track_name']) & \
               (df_all_updated['album_name'] == row['album_name'])
        df_all_updated.loc[mask, 'energy'] = row['energy']
        df_all_updated.loc[mask, 'valence'] = row['valence']
    
    # Replace the table with updated data
    conn.execute("CREATE OR REPLACE TABLE final_analytical_set AS SELECT * FROM df_all_updated")

    # 4. Save predictions to separate table for reference
    conn.execute("CREATE OR REPLACE TABLE final_predictions AS SELECT * FROM results_df")

    print("\n--- FINAL BOSS COMPLETE ---")
    print(f"✅ Predicted energy/valence for {len(results_df)} tracks")
    print(f"✅ Updated final_analytical_set with AI-generated stats")
    
    # Verify final count
    final_count = conn.execute("SELECT COUNT(*) FROM final_analytical_set WHERE energy IS NOT NULL").fetchone()[0]
    print(f"✅ Total tracks with energy/valence: {final_count}")
    
    conn.close()


if __name__ == "__main__":
    train_and_predict()
