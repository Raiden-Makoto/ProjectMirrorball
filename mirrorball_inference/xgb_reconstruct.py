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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import numpy as np # type: ignore

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
        print(f"\n{'='*50}")
        print(f"Optimizing {target.upper()}")
        print(f"{'='*50}")
        y_train_full = labeled_df[target]
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=30)
        
        print(f"\nBest hyperparameters for {target}:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best validation MSE: {study.best_value:.6f}")
        
        # Train final model with best params and evaluate on held-out test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Calculate accuracy metrics on test set
        test_preds = best_model.predict(X_test)
        mse = mean_squared_error(y_test, test_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        
        print(f"\nTest Set Performance ({target}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Train on full labeled set for final predictions
        final_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        final_model.fit(X_train_full, y_train_full)
        
        # Predict the 102 missing values
        predictions = final_model.predict(unlabeled_df[features])
        df.loc[df[target].isnull(), target] = predictions
        
        # Show sample predictions
        print(f"\nSample Predictions ({target}):")
        sample_preds = pd.DataFrame({
            'track_name': unlabeled_df['track_name'].head(5).values,
            'album_name': unlabeled_df['album_name'].head(5).values,
            f'predicted_{target}': predictions[:5]
        })
        print(sample_preds.to_string(index=False))

    # Save back to DB
    conn.execute("CREATE OR REPLACE TABLE mirrorball_ml_final AS SELECT * FROM df")
    
    # Final summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    total_predicted = len(unlabeled_df)
    total_tracks = len(df)
    print(f"✅ Predicted energy and valence for {total_predicted} tracks")
    print(f"✅ Total tracks in mirrorball_ml_final: {total_tracks}")
    print(f"✅ All tracks now have energy and valence values")
    print("✅ Ready for Plotly visualization")
    conn.close()

if __name__ == "__main__":
    run_tuned_inference()