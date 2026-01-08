"""
XGBoost Model for Reconstructing Valence and Energy Scores.

This module implements the final data cleaning workflow:
1. Train XGBoost on 231 labeled tracks (with energy/valence)
2. Use reading_grade, lexical_diversity, bridge_shift as features
3. Predict energy and valence for 102 unlabeled tracks
4. Update final_analytical_set with AI-generated predictions
"""

import os

import duckdb  # type: ignore
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
import optuna # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

def run_tuned_inference() -> None:
    conn = duckdb.connect(DB_PATH) # type: ignore
    df = conn.execute("SELECT * FROM final_analytical_set").df()
    
    # Features (Reading Grade, Bridge Shift, etc.)
    features = ['reading_grade', 'syllable_density', 'lexical_diversity', 'bridge_shift']
    
    # Split into Labeled (231) and Unlabeled (102)
    labeled_df = df[df['energy'].notnull()]
    unlabeled_df = df[df['energy'].isnull()]
    
    X_train_full = labeled_df[features]
    
    # Run Optuna for Energy and Valence
    for target in ['energy', 'valence']:
        print(f"--- Optimizing {target.upper()} ---")
        y_train_full = labeled_df[target]
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=30)
        
        # Train final model with best params
        best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train_full, y_train_full)
        
        # Predict the 102 missing values
        df.loc[df[target].isnull(), target] = best_model.predict(unlabeled_df[features])

    # Save back to DB
    conn.execute("CREATE OR REPLACE TABLE mirrorball_ml_final AS SELECT * FROM df")
    print("102 tracks 'reconstructed' using Tuned XGBoost. Ready for Plotly.")
    conn.close()

if __name__ == "__main__":
    run_tuned_inference()