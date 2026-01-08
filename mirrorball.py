import os
import duckdb # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import umap # type: ignore
import shap # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

def ship_high_quality_results() -> None:
    conn = duckdb.connect(DB_PATH) # type: ignore
    
    # 1. DATA EXTRACTION
    df = conn.execute("""
        SELECT track_name, album_name, 
               energy, valence, tempo, 
               bridge_sentiment_shift, reading_grade, 
               syllable_density, lexical_diversity, rhyme_density
        FROM final_mirrorball_dataset
    """).df().dropna()

    features = ['energy', 'valence', 'tempo', 'bridge_sentiment_shift', 
                'reading_grade', 'syllable_density', 'lexical_diversity', 'rhyme_density']
    
    # 2. SCALING to prevent feature dominance
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])
    
    # 3. K-MEANS to cluster the tracks into 5 'Sonic Eras'
    # We'll use k=5 for the 5 'Sonic Eras'
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(x_scaled)
    
    # 4. UMAP to create a 2D embedding of the tracks
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(x_scaled)
    df['umap_x'], df['umap_y'] = embedding[:, 0], embedding[:, 1]
    
    # 5. SHAP to explain the clusters
    # We use a KernelExplainer to see how features push songs into clusters
    explainer = shap.KernelExplainer(kmeans.predict, shap.sample(x_scaled, 50))
    shap_values = explainer.shap_values(x_scaled)
    
    # Add top contributing feature to each row for the Plotly Hover
    top_feature_idx = np.abs(shap_values).argsort()[-1]
    df['top_driver'] = [features[idx] for idx in top_feature_idx]

    # 6. SAVE TO DB
    conn.execute("CREATE OR REPLACE TABLE mirrorball_ml_final AS SELECT * FROM df")
    
    print("\n--- HIGH QUALITY ML RESULTS SHIPPED ---")
    print(df[['track_name', 'cluster_id', 'top_driver']].head(10))
    conn.close()

if __name__ == "__main__":
    ship_high_quality_results()