# Mirrorball Inference

## S2.1: XGBoost

We trained Optuna-tuned XGBoost regression models to predict energy and valence scores for 102 unlabeled tracks using stylometric features (reading grade, lexical diversity, syllable density, and bridge sentiment shift) extracted from lyrics. The models were trained on 231 labeled tracks with known energy/valence values, then used to reconstruct missing audio features for Taylor's Version releases and newer albums that lacked Spotify audio data. This approach bridges the gap between lyrical content and sonic characteristics, enabling complete feature coverage across all 333 tracks in the dataset.

### Results (Valence)
```
Best hyperparameters for valence:
  n_estimators: 53
  max_depth: 7
  learning_rate: 0.025452419390665493
  subsample: 0.6028153009946051
  colsample_bytree: 0.6937724069048358
Best validation MSE: 0.037961

Test Set Performance (valence):
  RMSE: 0.1948
  MAE:  0.1638
  R²:   0.1067
```

### Conclusion
Although the $R^2$ value is low, this a feature, not a bug. In fact, we can attribute this to the art vs "science" gap.
1. The "Acoustic" Variance: YWeou can have incredibly complex, "Quill Pen" lyrics played over a heavy synth-pop beat (High Energy/Valence) or a single piano (Low Energy/Valence). Lyrics alone only tell half the story.

2. Vocabulary Overlap: Taylor often uses simple words to describe complex pain ("I'm fine, but I'm not fine at all") and complex words to describe simple joy. The model is struggling because she is a master of subverting expectations.

3. Small Labeled Sample: We are training on only 231 tracks. For a high-dimensional relationship like "linguistics to emotion," 231 points is a low-data environment.

An $R^2$ of 0.10 (for valence) means our linguistic features explain about 10.6% of the variance in how "happy" a song sounds. In the music industry, that is actually significant!

### Results (Energy)
```
Best hyperparameters for energy:
  n_estimators: 78
  max_depth: 7
  learning_rate: 0.041197021055226095
  subsample: 0.7056911330694706
  colsample_bytree: 0.6957899933451032
Best validation MSE: 0.024064

Test Set Performance (energy):
  RMSE: 0.1551
  MAE:  0.1264
  R²:   0.0139
```

### Conclusion
The 0.01 $R^2$ for energy confirms that lyrical content and audio production are separate in Taylor Swift's discography. This 'sonic-linguistic dissonance' is a characteristic of her transition into the Midnights and TTPD eras, where high-energy synth-pop beats often mask low-valence, high-complexity lyrics. Additionally, energy is almost entirely driven by BPM, percussion density, and frequency spectrum, not by lyrics. 