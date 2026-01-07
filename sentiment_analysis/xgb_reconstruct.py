"""
XGBoost Model for Reconstructing Valence and Energy Scores.

This module trains XGBoost regression models to predict valence and energy scores
for new Taylor Swift tracks based on NLP features extracted from lyrics.
The models are trained on legacy tracks with known valence/energy values,
then used to predict scores for new era tracks.
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

# Feature columns used for training
FEATURES: List[str] = [
    "sentiment_compound",
    "lexical_complexity",
    "word_count",
    "sentiment_pos",
    "sentiment_neg",
]

# Target variables to predict
TARGETS: List[str] = ["valence", "energy"]


def train_and_predict() -> None:
    """
    Train XGBoost models and generate predictions for new era tracks.

    This function:
    1. Loads training data from master_training_data table
    2. Loads new era tracks that need predictions
    3. Trains separate XGBoost models for valence and energy
    4. Validates models using MAE (Mean Absolute Error)
    5. Generates predictions for new tracks
    6. Saves results to final_predictions and mirrorball_final_export tables

    The final export table combines actual (from legacy_tracks) and predicted
    (from ML model) values for visualization in dashboards.
    """
    conn = duckdb.connect(DB_PATH)

    # 1. Load Training Data
    print("Loading training data...")
    df_train = conn.execute("SELECT * FROM master_training_data").df()

    # 2. Load "New Era" Data (Tracks needing prediction)
    # We join lyrics features with the list of tracks that weren't in the legacy CSV
    print("Loading new era tracks for prediction...")
    df_new = conn.execute(
        """
        SELECT n.* 
        FROM dim_nlp_features n
        LEFT JOIN legacy_tracks l ON LOWER(n.track_name) = LOWER(l.track_name)
        WHERE l.track_name IS NULL
        """
    ).df()

    # Initialize results dataframe with track metadata
    results_df = df_new[["track_name", "album_name"]].copy()

    # Train a separate model for each target variable
    for target in TARGETS:
        print(f"\n--- Training Model for {target.upper()} ---")

        X = df_train[FEATURES]
        y = df_train[target]

        # Split for validation (80/20 train/test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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

        # Predict on New Eras
        results_df[target] = model.predict(df_new[FEATURES])

    # 3. Save Final Predictions to DB
    print("\nSaving predictions to database...")
    conn.execute("CREATE OR REPLACE TABLE final_predictions AS SELECT * FROM results_df")

    # 4. Create the "Tableau Export" Table
    # This combines legacy data + new predictions into one clean format
    conn.execute(
        """
        CREATE OR REPLACE TABLE mirrorball_final_export AS
        SELECT track_name, album_name, valence, energy, 'ACTUAL' as data_type 
        FROM legacy_tracks
        UNION ALL
        SELECT track_name, album_name, valence, energy, 'PREDICTED' as data_type 
        FROM final_predictions
        """
    )

    print("\n--- INFERENCE COMPLETE ---")
    print(f"Stats reconstructed for {len(results_df)} new tracks.")
    conn.close()


if __name__ == "__main__":
    train_and_predict()
