
# LatentPitch: High-Dimensional Latent Profiling & Scouting Architecture
A football player similarity and profiling tool built on high-dimensional latent space analysis. It standardises per-90 event metrics across the top five European leagues, reduces them via PCA, and measures similarity between players using cosine distance in the resulting latent space.

## Overview

LatentPitch takes raw player telemetry, normalises it for playing time, runs positional cohort-conditioned PCA, and returns a ranked list of the most similar players to any chosen target. The default configuration uses 10 principal components, capturing ~83.7% of active metric variance, and operates on a 171-feature dataset for the 2025/26 season.

Two model frameworks are available:
- **Season-specific (undiluted):** PCA fitted strictly on the active season, preserving full matrix rank.
- **Multi-season (global):** PCA fitted across a unified three-season horizon for broader career-arc comparisons.

An in-app manuscript tab details the full methodology, design decisions, and known limitations.

---
## System Architecture

```
Raw data (CSV)
    └── data_loader.py
            Per-90 normalisation
            Positional cohort imputation (median within role)
            450-minute minimum threshold filter
            └── Parquet cache
                    └── pca_engine.py
                            Z-score standardisation within cohort
                            PCA (k = 7, 10, or 12 components)
                            Cosine similarity ranking
                            └── app.py (Streamlit UI)
```


## Data Sources

| Dataset | Season | Curator | Features |
| --- | --- | --- | --- |
| Dataset B (primary) | 2025/26 | Chuong Trinh (Kaggle) | 171 |
| Dataset A (legacy) | 2025/26 | Hubert Sidorowicz (Kaggle) | 102 |
| Historical | 2024/25 | Hubert Sidorowicz (Kaggle) | 267 |
| Forward-looking | 2026/27 | Hubert Sidorowicz (Kaggle) | 102 |

Dataset B is the primary active dataset. It includes expected goals (xG), expected assists (xAG), duel win rates, dribbles, recoveries, and final-third passing metrics not present in Dataset A.


## Installation

```bash
git clone https://github.com/AZ-Meriem/LatentPitch.git
cd LatentPitch
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Pre-built parquet caches are included; run `python data_loader.py` only if you need to regenerate them from the raw CSVs.

---

## Project Structure

```
LatentPitch/
├── app.py                  # Streamlit UI
├── data_loader.py          # ETL pipeline and parquet generation
├── pca_engine.py           # PCA reduction and cosine similarity
├── requirements.txt
├── LICENSE
├── README.md
├── Data/
│   ├── PERFECT_players_data-2025_2026.csv   # Dataset B (171 cols)
│   ├── players_data-2024_2025.csv
│   ├── players_data-2025_2026.csv
│   ├── players_data-2026_2027.csv
│   ├── processed_players_dataset_b.parquet
│   └── processed_players_dataset_a.parquet
├── benchmarks/             # Validation and audit scripts
└── before_after_comparison/ # Historical similarity audit logs
```

---

## Methodology

1. **Per-90 scaling** — removes playing-time bias.
2. **Cohort imputation** — missing values filled with the positional cohort median.
3. **Cohort isolation** — goalkeepers and outfield players run in separate latent spaces.
4. **Z-score standardisation** — zero mean, unit variance within each cohort.
5. **PCA** — covariance matrix decomposed into k orthogonal components (k = 7, 10, or 12).
6. **Cosine similarity** — similarity between two latent vectors, expressed as a percentage.

Variance explained: 7 PCs = 76.6%, 10 PCs = 83.7%, 12 PCs = 87.4%. 10 PCs is the recommended default.

---


## Limitations

- Tactical and possession bias: players in high-possession sides accumulate higher pass and touch volumes.
- No cross-league strength adjustment.
- On-ball events only — pressing runs, off-ball movement, and defensive shape are not captured.
- PCA is linear; non-linear metric interactions are not explicitly modelled.
- 450-minute minimum threshold excludes short-stint players and late-window signings.

---


## Project Contribution and AI Collaboration

**Technical Director:** Lead researcher (`z3757181@gmail.com`) — responsible for the research hypothesis, dual-engine architecture design, dataset selection, empirical validation, and all analytical decisions throughout the project.

**AI implementation:** Antigravity (Google DeepMind) was used as a pair-programming assistant for code generation, refactoring, and pipeline implementation under the Technical Director's direction.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

MIT. See [LICENSE](LICENSE).
