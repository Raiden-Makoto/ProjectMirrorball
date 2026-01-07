"""
Bridge Impact Analysis for Project Mirrorball.

This module analyzes the emotional impact of bridges in Taylor Swift's songs.
It extracts bridge sections from lyrics, calculates sentiment shifts between
verses and bridges, and adds these features to the master training dataset.

Taylor Swift is known as the "Bridge Queen" for her impactful bridge sections,
so this analysis captures that stylistic signature.
"""

import os
from typing import Dict, Optional

import duckdb  # type: ignore
import pandas as pd  # type: ignore
import re
from tqdm import tqdm  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mirrorball.db")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


def analyze_bridge_dynamics(lyrics: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Analyze bridge dynamics by extracting verse, bridge, and chorus sections.

    This function:
    1. Parses lyrics to identify verse, bridge, and chorus sections using tags
    2. Calculates sentiment scores for each section using VADER
    3. Computes bridge sentiment shift (bridge - verse)
    4. Computes bridge-chorus contrast (bridge - chorus)

    Args:
        lyrics: Full lyrics text with section tags. Can be None or empty.

    Returns:
        Dictionary with bridge metrics:
        - bridge_sentiment: Absolute sentiment score of the bridge
        - bridge_sentiment_shift: Difference between bridge and verse sentiment
        - bridge_chorus_contrast: Difference between bridge and chorus sentiment
        - has_bridge: 1 if bridge exists, 0 otherwise
        - bridge_word_count: Number of words in the bridge
        Returns None if lyrics is empty.
    """
    if not lyrics:
        return None

    # 1. Split by tags but keep the tags to identify sections
    # Regex splits at [Verse], [Bridge], etc.
    parts = re.split(r"(\[.*?\])", lyrics)

    sections = {"verse": "", "bridge": "", "chorus": ""}
    current_label = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("[") and part.endswith("]"):
            label = part.lower()
            if "verse" in label:
                current_label = "verse"
            elif "bridge" in label:
                current_label = "bridge"
            elif "chorus" in label:
                current_label = "chorus"
            else:
                current_label = "other"
        elif current_label in sections:
            sections[current_label] += " " + part

    # 2. Sentiment Analysis per section
    # We use conditional checks to handle cases where a section might be missing
    v_sent = (
        analyzer.polarity_scores(sections["verse"])["compound"]
        if sections["verse"]
        else 0
    )
    b_sent = (
        analyzer.polarity_scores(sections["bridge"])["compound"]
        if sections["bridge"]
        else 0
    )
    c_sent = (
        analyzer.polarity_scores(sections["chorus"])["compound"]
        if sections["chorus"]
        else 0
    )

    # 3. Feature Engineering
    # The 'Shift' is how much the bridge deviates from the setup (Verse)
    bridge_shift = b_sent - v_sent

    # The 'Contrast' is how different the bridge is from the main hook (Chorus)
    bridge_contrast = b_sent - c_sent

    return {
        "bridge_sentiment": b_sent,
        "bridge_sentiment_shift": bridge_shift,
        "bridge_chorus_contrast": bridge_contrast,
        "has_bridge": 1 if sections["bridge"] else 0,
        "bridge_word_count": len(sections["bridge"].split()),
    }


def main() -> None:
    """
    Main function to process all tracks and calculate bridge impact metrics.

    This function:
    1. Loads all lyrics from dim_lyrics table
    2. Analyzes bridge dynamics for each track
    3. Saves metrics to dim_bridge_metrics table
    4. Reports the most dramatic bridge shift found
    """
    conn = duckdb.connect(DB_PATH)

    # Fetch the newly rescraped lyrics
    df = conn.execute(
        "SELECT track_name, album_name, lyrics FROM dim_lyrics"
    ).df()

    bridge_results = []
    print("Running Bridge Impact Analysis...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        metrics = analyze_bridge_dynamics(row["lyrics"])
        if metrics:
            metrics.update(
                {
                    "track_name": row["track_name"],
                    "album_name": row["album_name"],
                }
            )
            bridge_results.append(metrics)

    # Convert to DataFrame and update database
    res_df = pd.DataFrame(bridge_results)
    conn.execute(
        "CREATE OR REPLACE TABLE dim_bridge_metrics AS SELECT * FROM res_df"
    )

    # Report completion
    print("Bridge Metrics calculated and saved to 'dim_bridge_metrics'")

    # Let's find the 'Most Dramatic' bridge in the dataset
    top_shift = conn.execute(
        """
        SELECT track_name, album_name, bridge_sentiment_shift 
        FROM dim_bridge_metrics 
        ORDER BY ABS(bridge_sentiment_shift) DESC 
        LIMIT 1
    """
    ).fetchone()

    if top_shift:
        print(
            f"Most Dramatic Bridge Shift: {top_shift[0]} ({top_shift[1]}) with {round(top_shift[2], 3)} shift"
        )

    conn.close()


if __name__ == "__main__":
    main()
