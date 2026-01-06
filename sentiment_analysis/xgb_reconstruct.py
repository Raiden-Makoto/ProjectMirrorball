import os
import duckdb # type: ignore
import pandas as pd # type: ignore
import xgboost as xgb # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore, use squared=False for RMSE

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def train_and_predict():
    conn = duckdb.connect(DB_PATH)
    
    # 1. Load Training Data
    df_train = conn.execute("SELECT * FROM master_training_data").df()
    
    # 2. Load "New Era" Data (Tracks needing prediction)
    # We join lyrics features with the list of tracks that weren't in the legacy CSV
    df_new = conn.execute("""
        SELECT n.* FROM dim_nlp_features n
        LEFT JOIN legacy_tracks l ON LOWER(n.track_name) = LOWER(l.track_name)
        WHERE l.track_name IS NULL
    """).df()

    features = ['sentiment_compound', 'lexical_complexity', 'word_count', 'sentiment_pos', 'sentiment_neg']
    targets = ['valence', 'energy']
    
    results_df = df_new[['track_name', 'album_name']].copy()

    for target in targets:
        print(f"\n--- Training Model for {target.upper()} ---")
        
        X = df_train[features]
        y = df_train[target]
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost Regressor optimized for Apple Silicon
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            tree_method="hist", # Efficient on CPU
            device="cpu"
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model Accuracy (MAE): {round(mae, 4)}")
        
        # Predict on New Eras
        results_df[target] = model.predict(df_new[features])

    # 3. Save Final Predictions to DB
    conn.execute("CREATE OR REPLACE TABLE final_predictions AS SELECT * FROM results_df")
    
    # 4. Create the "Tableau Export" Table
    # This combines legacy data + new predictions into one clean format
    conn.execute("""
        CREATE OR REPLACE TABLE mirrorball_final_export AS
        SELECT track_name, album_name, valence, energy, 'ACTUAL' as data_type FROM legacy_tracks
        UNION ALL
        SELECT track_name, album_name, valence, energy, 'PREDICTED' as data_type FROM final_predictions
    """)
    
    print("\n--- INFERENCE COMPLETE ---")
    print(f"Stats reconstructed for {len(results_df)} new tracks.")
    conn.close()

if __name__ == "__main__":
    train_and_predict()