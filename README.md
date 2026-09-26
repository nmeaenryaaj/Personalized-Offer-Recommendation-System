# American Express Campus Challenge 2025 — Offer Ranking Model

**Team:** Mohammad Umam Ali, Polaki Snehitha, Neeraj Sharma
**Institution:** IIT Guwahati (Skills Matters)

## Problem Statement

Given user, offer, transaction, and event-level data, predict and rank the offers each user is most likely to engage with (top-7 ranking task, evaluated on **MAP@7**). The core challenge was severe **feature drift** between the training data and the unseen/leaderboard data, which caused naive high-performing models to collapse out of sample.

## Approach Summary

The pipeline is built in three stages: leak-proof feature engineering, aggressive drift-based feature selection, and a stacked ranking ensemble.

### 1. Feature Engineering

Features were engineered from three data sources:

- **Train data:** debit totals over 30d/180d windows, time-since-last-event, time-since-last-system-event, time-band buckets
- **Offers & transactions:** discount rate, NLP-based offer-text embeddings and similarity scores
- **Events data:** offer impressions/clicks over 1h/3h/6h/9h windows, historical and daywise CTR

**Text feature engineering (offer descriptions):**
1. Clean and chronologically sort offer text per user session
2. Vectorize with a `SentenceTransformer` (384-dim embeddings)
3. Reduce to 32 dimensions via PCA and unpack into individual columns
4. Compute cosine similarity between an offer and the user's live session centroid (session similarity), which showed a clear upward trend as sessions progress — later clicks are more session-focused

Offer IDs were also **target-encoded** based on historical performance.

### 2. Feature Cleaning & Selection

A multi-stage filtering process was used to keep the model robust against drift:

| Step | Method | Result |
|---|---|---|
| Cleaning | Dropped columns with >90% missing values | 26 columns removed |
| Cleaning | Dropped zero-variance columns (std = 0) | 59 columns removed |
| Cleaning | `log1p` transform on skewed numeric features (based on mean/median imbalance) | normalized distributions |
| Cleaning | Mean imputation (numeric), mode imputation (categorical) | — |
| Drift detection (univariate) | KS-test (numeric, stat > 0.1) and chi-squared test (categorical, p < 0.05) between train and unseen data | 119 features removed (97 numeric + 22 categorical) |
| Drift detection (multivariate) | Adversarial validation — binary classifier trained to distinguish train vs. unseen rows | AUC > 0.7 flagged strong drift; 45+ high-drift features removed |
| Final selection | XGBRanker feature importances used to validate and prune engineered features | Reduced to the **top 15** most predictive features |

Adversarial AUC dropped from **1.0 → 0.82** after drift-feature removal, which is what stabilized leaderboard performance (see Results).

### 3. Modeling

**Base ranker (`rank:ndcg`):**
- XGBoost Ranker trained on grouped/labeled query data, directly optimizing NDCG via lambda-gradients
- Achieved **MAP@7 ≈ 0.71** on a time-series validation split, but collapsed to **0.56** on the leaderboard due to feature drift
- After adversarial-validation-based feature pruning, leaderboard MAP@7 recovered to **~0.53**

**Stacking ensemble (final model):**
- Three base rankers trained in parallel:
  - **XGBoost** — `rank:ndcg`
  - **LightGBM** — `lambdarank`
  - **CatBoost** — `YetiRank`
- Raw rank scores from each model used as features for a **logistic regression meta-model**, which outputs final probabilities
- An **ablation study** showed LightGBM dominated the stack (highest meta-model weight), XGBoost added a minor lift, and CatBoost's contribution was negligible
- **Final model = LightGBM + XGBoost** (CatBoost dropped after ablation) — a leaner ensemble with no meaningful loss in performance

## Results

| Model | Relative MAP@7 (out-of-sample) |
|---|---|
| Base model (before adversarial selection) | Lowest |
| Base model (after adversarial selection) | Dip (traded stability for drift-robustness) |
| Base model + feature engineering | Recovered |
| LightGBM (standalone) | Strong |
| CatBoost (standalone) | Strong |
| Full ensemble (XGB + LGBM + CatBoost) | Best |
| **Final ensemble (XGB + LGBM only)** | **Best, most efficient** |

The final stacked ensemble delivered a **~5% MAP@7 improvement** over the single best base model (LightGBM) while being more robust to feature drift than any individual model.

## Repository Contents

- `notebooks/` — exploratory data analysis, feature engineering, and modeling notebooks
- `Amex_DS_Campus_Challenge_Deck.pptx` — final presentation deck summarizing the approach and results

## Future Work

- **Graph-based features:** build a user–offer interaction graph and apply Node2Vec-style embeddings to capture latent relationships
- **Cross-domain features:** engineer explicit correlations across transaction, offer, and event data
- **Field-aware Factorization Machines (FFMs)** for more granular pairwise feature interactions in CTR prediction
- **Behavioral clustering:** segment users (e.g. "High-Frequency Shoppers", "Travel Enthusiasts") and train specialized ranking models per segment
- **Cold-start model:** a dedicated model for new offers with no historical data, trained purely on offer metadata

## Challenges

- Highly imbalanced dataset with no obvious initial modeling path
- **DeepFM** was explored but proved highly sensitive to feature drift and required extensive tuning/regularization, making it less robust than the final tree-based ensemble

## References

- Rendle, S. et al. — Field-aware Factorization Machines: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
- LIBFFM implementation: https://github.com/ycjuan/libffm
- Light-FFM: https://github.com/daxiongshu/light-ffm
- https://doi.org/10.55524/ijirem.2025.12.1.4
