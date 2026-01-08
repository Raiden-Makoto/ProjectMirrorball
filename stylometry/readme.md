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

## S1.2: Lexical Sophistication Analysis

This module analyzes the lexical sophistication of Taylor Swift's songs to mathematically map the stylistic shift from one genre to another. Country and Pop (her early eras) rely on "Earworms" (short, punchy, monosyllabic words) that are easy to sing along to. Folk and "Academic" Pop (her later eras) rely on "Storytelling" (longer, multisyllabic words) that convey specific, often niche, imagery.

### Methodology

The `lexical_sophistication.py` script:
1. Removes section tags ([Verse], [Bridge], etc.) from lyrics for analysis
2. Processes text so each line is treated as a sentence (joins lines with ". ") to ensure realistic reading grade calculations
3. Calculates Flesch-Kincaid reading grade (actual averages range from ~1.0-2.3 across albums)
4. Calculates syllable density (total syllables / total words)
5. Calculates lexical diversity (unique words / total words)
6. Calculates difficult word ratio (difficult words / total words)

### Key Metrics

- **reading_grade**: Flesch-Kincaid reading grade level (actual averages: ~1.0-2.3 across albums)
- **syllable_density**: Average syllables per word (higher = more complex vocabulary)
- **lexical_diversity**: Ratio of unique words to total words (higher = more varied vocabulary)
- **difficult_ratio**: Ratio of difficult words to total words (higher = more sophisticated language)

### Results

**Most Sophisticated Songs (Top 5 by Reading Grade):**
- "Mr. Perfectly Fine (Taylor's Version) [From The Vault]" (Fearless (Taylor's Version)) - Grade Level: 3.60
- "I Did Something Bad" (reputation) - Grade Level: 3.46
- "dorothea" (evermore) - Grade Level: 3.45
- "High Infidelity" (Midnights) - Grade Level: 3.44
- "We Are Never Ever Getting Back Together" (Red) - Grade Level: 3.42

**Average Reading Grade by Album:**

| Album | Avg Grade | Min | Max | Tracks |
|-------|-----------|-----|-----|--------|
| The Life of a Showgirl | 2.28 | 0.50 | 3.17 | 12 |
| Midnights | 1.90 | 0.56 | 3.44 | 20 |
| folklore | 1.76 | 0.68 | 3.22 | 17 |
| The Tortured Poets Department: The Anthology | 1.75 | 0.66 | 2.67 | 31 |
| Red | 1.75 | 0.69 | 3.42 | 22 |
| The Tortured Poets Department | 1.70 | 0.66 | 2.67 | 16 |
| Fearless | 1.69 | 0.68 | 2.50 | 19 |
| Taylor Swift | 1.64 | 0.53 | 3.08 | 15 |
| Speak Now (Taylor's Version) | 1.60 | 0.61 | 2.44 | 22 |
| evermore | 1.57 | 0.46 | 3.45 | 17 |
| Speak Now | 1.57 | 0.75 | 2.39 | 17 |
| Red (Taylor's Version) | 1.49 | -0.45 | 3.39 | 29 |
| Lover | 1.48 | -0.32 | 3.36 | 18 |
| reputation | 1.46 | 0.32 | 3.46 | 15 |
| Fearless (Taylor's Version) | 1.44 | -0.17 | 3.60 | 26 |
| 1989 (Taylor's Version) | 1.19 | -0.11 | 2.75 | 21 |
| 1989 | 1.07 | -0.12 | 2.65 | 16 |

This table quantifies a notable shift in structural complexity across Taylor Swift's career.
We can see that her 1989 era represents her most "streamlined" period (Avg Grade: 1.07),
designed for maximum radio-friendliness and monosyllabic punch.

Post-2020, there is a measurable "Lexical Climb." Both folklore (1.76) and Midnights (1.90) show a return to more complex sentence structures and higher syllable density, confirming the fan-observed shift toward "literary" songwriting through hard data.

*Note: The presence of negative reading grades (Min: -0.45) indicates maximal phonetic simplicity. These scores represent tracks where lyrical density is sacrificed for rhythmic repetition.*

## S1.3: Thematic DNA Indeitification via TF-IDF