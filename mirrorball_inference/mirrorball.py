import os
import duckdb # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import umap # type: ignore
import shap # type: ignore
import xgboost as xgb # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def mirrorball_inference() -> None:
    conn = duckdb.connect(DB_PATH) # type: ignore
    
    # 1. UNIFY DATA (Join all your engineered features)
    df = conn.execute("""
        SELECT 
            l.track_name, l.album_name, 
            m.energy, m.valence,
            lx.reading_grade, lx.syllable_density, lx.lexical_diversity,
            COALESCE(b.bridge_sentiment_shift, 0) as bridge_shift,
            m.thematic_dna
        FROM dim_lyrics l
        LEFT JOIN master_training_data m ON l.track_name = m.track_name AND l.album_name = m.album_name
        LEFT JOIN dim_lexical_metrics lx ON l.track_name = lx.track_name AND l.album_name = lx.album_name
        LEFT JOIN dim_bridge_metrics b ON l.track_name = b.track_name AND l.album_name = b.album_name
    """).df()

    features = ['reading_grade', 'syllable_density', 'lexical_diversity', 'bridge_shift']
    
    # 2. XGBOOST INFERENCE (Fill the 102 missing tracks using your Optuna params)
    labeled = df[df['energy'].notnull()]
    unlabeled = df[df['energy'].isnull()]
    
    best_params = {
        "energy": {"n_estimators": 53, "max_depth": 7, "learning_rate": 0.025},
        "valence": {"n_estimators": 53, "max_depth": 7, "learning_rate": 0.025}
    }

    for target in ['energy', 'valence']:
        # Track which rows will be predicted BEFORE filling them
        df[f'{target}_is_predicted'] = df[target].isnull()
        model = xgb.XGBRegressor(**best_params[target], random_state=42)
        model.fit(labeled[features], labeled[target])
        df.loc[df[target].isnull(), target] = model.predict(unlabeled[features])

    # 3. K-MEANS & UMAP (The Latent Space)
    all_features = features + ['energy', 'valence']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[all_features])
    
    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(x_scaled)
    
    # Dimensionality Reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(x_scaled)
    df['umap_x'], df['umap_y'] = embedding[:, 0], embedding[:, 1]

    # 4. SHAP (Explain the Clusters)
    # We explain why the machine put a song in its specific cluster
    explainer = shap.KernelExplainer(kmeans.predict, shap.kmeans(x_scaled, 10))
    shap_vals = explainer.shap_values(x_scaled)
    
    # Identify the Top Driver for each song
    # SHAP Transparency: Prefer linguistic drivers over energy/valence
    # (since R² was only 1-10%, energy/valence aren't reliable predictors)
    linguistic_features = ['reading_grade', 'syllable_density', 'lexical_diversity', 'bridge_shift']
    audio_features = ['energy', 'valence']
    
    top_drivers = []
    for i, val in enumerate(shap_vals):
        abs_impacts = np.abs(val)
        top_idx = np.argmax(abs_impacts)
        top_feature = all_features[top_idx]
        
        # If top driver is energy/valence, check if a linguistic feature has significant impact
        if top_feature in audio_features:
            # Find the top linguistic feature
            linguistic_indices = [all_features.index(f) for f in linguistic_features if f in all_features]
            if linguistic_indices:
                linguistic_impacts = abs_impacts[linguistic_indices]
                top_linguistic_idx = linguistic_indices[np.argmax(linguistic_impacts)]
                top_linguistic_impact = abs_impacts[top_linguistic_idx]
                
                # Use linguistic feature if it's at least 70% of the audio feature's impact
                if top_linguistic_impact >= 0.7 * abs_impacts[top_idx]:
                    top_drivers.append(all_features[top_linguistic_idx])
                else:
                    top_drivers.append(top_feature)
            else:
                top_drivers.append(top_feature)
        else:
            top_drivers.append(top_feature)
    
    df['top_driver'] = top_drivers

    # 5. FINAL EXPORT
    conn.execute("CREATE OR REPLACE TABLE final_map_data_with_shap AS SELECT * FROM df")
    print("PROJECT COMPLETED: 333 tracks processed with SHAP and UMAP.")
    print(df[['track_name', 'cluster_id', 'top_driver']].head(10))
    conn.close()

if __name__ == "__main__":
    mirrorball_inference()