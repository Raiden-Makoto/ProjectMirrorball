# Stylometry / Computational Linguistics Analysis of Taylor Swift's Songs

## S1.1: Bridge Impact Analysis

This module analyzes the emotional impact of bridges in Taylor Swift's songs, capturing her signature "Bridge Queen" style.

### Methodology

The `bridge_impact.py` script:
1. Extracts verse, bridge, and chorus sections from lyrics using section tags
2. Calculates VADER sentiment scores for each section
3. Computes bridge sentiment shift (bridge - verse) to measure emotional deviation
4. Computes bridge-chorus contrast (bridge - chorus) to measure hook contrast
5. Identifies tracks with bridges and counts bridge word count

### Key Metrics

- **bridge_sentiment**: Absolute sentiment score of the bridge section
- **bridge_sentiment_shift**: Emotional shift from verse to bridge (negative = sadder, positive = happier)
- **bridge_chorus_contrast**: Contrast between bridge and chorus sentiment
- **has_bridge**: Binary indicator if track contains a bridge section
- **bridge_word_count**: Number of words in the bridge section

### Results

- **305 tracks** (91.6%) contain bridge sections
- **Average bridge sentiment shift**: -0.0860 (slightly sadder on average)
- **Most dramatic negative shift**: "Cold As You" (-1.9162)
- **Most dramatic positive shift**: "New Romantics" (+1.8474)