![Project Mirrorball Latent Space](mirrorball.png)

Project Mirrorball applies computational linguistics and machiine learning to analyze Taylor Swift's songwriting across her entire career. The project extracts stylometric features from lyrics, uses ML to bridge gaps in audio feature data from the Spotify API, and visualizes the "latent space" of her discography, revealing how her songwriting has evolved from country storytelling to literary pop.

## Methodology

### Phase 1: Stylometric Feature Extraction

**S1.1: Bridge Impact Analysis** (`stylometry/bridge_impact.py`)
- Analyzes the emotional impact of bridges using VADER sentiment analysis
- Measures bridge sentiment shift (bridge - verse) and bridge-chorus contrast
- **Finding**: 305 tracks (91.6%) contain bridges with an average sentiment shift of -0.0860 (slightly sadder)

**S1.2: Lexical Sophistication** (`stylometry/lexical_sophistication.py`)
- Calculates Flesch-Kincaid reading grade, syllable density, and lexical diversity
- Maps the stylistic shift from "Earworms" (monosyllabic, radio-friendly) to "Storytelling" (multisyllabic, literary)
- **Finding**: 1989 era represents the most streamlined period (Avg Grade: 1.07), while post-2020 shows a "Lexical Climb" (folklore: 1.76, Midnights: 1.90)

**S1.3: Thematic DNA Identification** (`stylometry/thematic_dna.py`)
- Uses TF-IDF to identify signature words that define each track's unique lyrical identity
- Applies deep cleaning to remove metadata noise (brackets, artist names) and explicit blacklisting
- Extracts top 5 most characteristic words per track based on TF-IDF scores

### Phase 2: Machine Learning Inference

**S2.1: XGBoost Reconstruction** (`mirrorball_inference/xgb_reconstruct.py`)
- Trains Optuna-tuned XGBoost models on 231 labeled tracks to predict energy/valence
- Uses stylometric features (reading grade, lexical diversity, syllable density, bridge shift) as inputs
- Reconstructs missing audio features for 102 unlabeled tracks (Taylor's Version releases, newer albums)
- **Performance**: Valence R² = 0.1067, Energy R² = 0.0139 (low R² reflects the "art vs science" gap—lyrics alone explain ~10% of sonic characteristics)

**S2.2: Latent Space Mapping** (`mirrorball_inference/mirrorball.py`)
- Unifies all features into a single analytical set (333 tracks)
- Applies K-Means clustering (5 clusters) to identify "Sonic Archetypes"
- Uses UMAP for 2D dimensionality reduction (creating the "star map" visualization)
- Applies SHAP explainability to identify the top driver for each track's cluster assignment
- **SHAP Transparency**: Prefers linguistic drivers over energy/valence when impact is comparable (since R² was low)

### Phase 3: Visualization

**Interactive Latent Space Browser** (`app.py`)
- Creates an interactive Plotly visualization of the 333-track latent space
- Color-codes by ML archetype clusters (Quill Pen, Fountain Pen, Glitter Gel Pen, Revenge Anthem, Standard Pop)
- Shape-codes by album era
- Exports as standalone HTML (`index.html`) and CSV (`mirrorball.csv`)

## Data Pipeline

1. **Ingestion** (`database/ingest_new.py`, `database/legacy_tracks.py`): Scrapes lyrics from Genius API
2. **Feature Engineering** (`database/feateng.py`): Extracts NLP features and sentiment scores
3. **Stylometry Analysis** (`stylometry/*.py`): Computes bridge metrics, lexical sophistication, thematic DNA
4. **Unification** (`database/unify.py`): Joins all features into `final_analytical_set` (333 tracks)
5. **ML Inference** (`mirrorball_inference/*.py`): Predicts missing features, clusters, and explains with SHAP
6. **Visualization** (`app.py`): Generates interactive map and exports

## Key Findings

- **Bridge Impact**: 91.6% of tracks contain bridges, with an average negative sentiment shift (bridges are slightly sadder than verses)
- **Lexical Evolution**: Measurable shift from streamlined pop (1989: 1.07) to literary complexity (folklore: 1.76, Midnights: 1.90)
- **Sonic-Linguistic Dissonance**: Low R² values (1-10%) confirm that lyrical content and audio production are largely independent—complex lyrics can exist over simple beats, and vice versa
- **Thematic DNA**: TF-IDF successfully isolates signature words that define each track's unique identity, from literal name-dropping (debut) to high-concept imagery (later eras)

## Visualization

**Interactive Latent Space Browser** (`app.py`)
- Plotly-powered interactive visualization of all 333 tracks in 2D latent space
- **Color-coding**: ML archetype clusters (Quill Pen, Fountain Pen, Glitter Gel Pen, Revenge Anthem, Standard Pop)
- **Shape-coding**: Album eras (each album gets a unique marker shape)
- **Hover tooltips**: Track name, album, archetype name, vibe description, and top SHAP driver
- **Dual legends**: Separate legend groups for archetypes (colors) and eras (shapes)

The visualization reveals the "star map" of Taylor Swift's discography—showing how tracks cluster by stylistic similarity and how her songwriting has evolved across eras.