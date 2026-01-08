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

def mirrorball_inference() -> None:
    conn = duckdb.connect(DB_PATH) # type: ignore
    df = conn.execute("SELECT * FROM final_map_data").df()
    
    # 1. Prepare Features
    features = ['energy', 'valence', 'reading_grade', 'syllable_density', 'lexical_diversity', 'bridge_shift']
    x = df[features].values
    
    # Scale data for the model (SHAP needs the same input as the model)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # 2. Fit the 'Final' K-Means model
    # We use n_clusters=5 to represent the 'Sonic Archetypes'
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    df['cluster_id'] = kmeans.labels_

    # 3. THE SHAP MOVE
    # We use KernelExplainer to explain the cluster assignment
    # We use a summary of the data as the background to speed it up
    background = shap.kmeans(x_scaled, 10) 
    explainer = shap.KernelExplainer(kmeans.predict, background)
    shap_values = explainer.shap_values(x_scaled)

    # 4. Extract the 'Top Driver' per song
    # shap_values for a regressor is a list of arrays. We find the feature with max absolute impact.
    top_drivers = []
    for i in range(len(df)):
        # Get absolute impact of each feature for this specific row
        impacts = np.abs(shap_values[i])
        top_feature_idx = np.argmax(impacts)
        top_drivers.append(features[top_feature_idx])
    
    df['top_driver'] = top_drivers

    # 5. Save back to DuckDB
    conn.execute("CREATE OR REPLACE TABLE mirrorball_analyzed_data AS SELECT * FROM df")
    
    print("\n--- SHAP ANALYSIS COMPLETE ---")
    print(df[['track_name', 'cluster_id', 'top_driver']].head(10))
    conn.close()

if __name__ == "__main__":
    mirrorball_inference()