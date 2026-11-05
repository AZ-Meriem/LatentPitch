# ⚽ LatentPitch: High-Dimensional Latent Profiling & Scouting Architecture

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LatentPitch** is an advanced association football player similarity and profiling platform. By standardizing high-dimensional telemetry across the top five European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), performing cohort-conditioned orthogonal decomposition (PCA), and measuring directional alignment via cosine similarity, LatentPitch identifies tactical player affinities without the noise of counting stats or playing-time bias.

---

## 🚀 Key Highlights

* **Calibrated Latent Space (10 PCs Default)**: Decomposes 46+ active per-90 event features into 10 orthogonal principal components capturing **83.66%** of active system variance, balancing core archetypes with granular tactical traits (wide crossing, box threat, recovery duel volume).
* **Enriched 2025/2026 Event Telemetry**: Integrated with Chuong Trinh's 171-parameter dataset (`PERFECT_players_data-2025_2026.csv`), tracking expected goals ($xG$), expected assists ($xAG$), duels won, dribbles, recoveries, final-third passes, and goalkeeper prevention.
* **Dual-Architecture Latent Framework**:
  * **Season-Specific (Undiluted - Primary Default)**: Fits PCA strictly within the active season snapshot to preserve full mathematical rank and eliminate zero-fill dilution from missing features.
  * **General Multi-Season (Global)**: Standardizes and computes coordinates across the unified 3-season horizon for macro multi-campaign evaluation.
* **Streamlined UI with Advanced Controls**: Clean, intuitive primary controls with an expandable **⚙️ Advanced Engine Configuration** panel for technical parameter overrides.
* **Interactive Scouting Diagnostics**:
  * **Top Latent Matches**: Instant cosine similarity rankings with real-time match percentage.
  * **Direct Dimensionality Comparison**: Expander displaying 7 vs 10 vs 12 PC rankings side-by-side on any target player.
  * **Role-Tailored Polar Radar**: Forwards/wingers receive attacking metrics; midfielders receive progression/retention metrics; defenders receive clearance/tackle metrics.
  * **3D Latent Vector Projection**: Interactive 3D scatter plot of the cohort embedding space.
  * **Metric Correlation Quadrants & Heatmaps**: Per-90 scatter matrices with median baselines.
* **In-App Technical Manuscript**: Built-in research documentation detailing mathematical formulation, evolution history, empirical findings, and analytical limitations.

---

## 📐 System Architecture

```
Raw Telemetry Sources (FBref / Kaggle)
 │
 ├── PERFECT_players_data-2025_2026.csv (Chuong Trinh - 171 cols) [Dataset B]
 ├── players_data-2025_2026.csv (Hubert Sidorowicz - 102 cols)     [Dataset A]
 ├── players_data-2024_2025.csv (Hubert Sidorowicz - 267 cols)
 └── players_data-2026_2027.csv (Hubert Sidorowicz - 102 cols)
       │
       ▼
 ┌─────────────────────────────────────────────────────────┐
 │               data_loader.py (Data Ingestion)           │
 │  • Per-90 Volumetric Normalization (Metric / 90s)       │
 │  • Primary Pos Categorization (FW, MF, DF, GK)          │
 │  • Sample Threshold Filtering (Min >= 450 mins)         │
 │  • Cohort Median Imputation strictly within Role        │
 │  • Positional Percentile Precomputation (0-100)         │
 └─────────────────────────┬───────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
 processed_players_dataset_b.parquet   processed_players_dataset_a.parquet
 (Primary Enriched Telemetry)          (Standard Legacy Baseline)
       │
       ▼
 ┌─────────────────────────────────────────────────────────┐
 │               pca_engine.py (Latent Space)              │
 │  • Outfield vs GK Cohort Isolation                      │
 │  • Z-score Standardization within Cohort                │
 │  • Season-Specific (Undiluted) vs Global Models         │
 │  • Sign-Aligned PCA (k = 7, 10, 12 components)          │
 │  • Latent Cosine Similarity Computation                 │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │                  app.py (Streamlit UI)                  │
 │  • Minimalist Slate Dark Theme                          │
 │  • Dynamic Polar Radar (7D/10D/12D Latent Overlay)      │
 │  • 3D Orthogonal Component Scatter                      │
 │  • Side-by-side Dimensionality Comparison               │
 │  • In-App Research Manuscript & Diagnostics             │
 └─────────────────────────────────────────────────────────┘
```

---

## 📦 Installation & Quickstart

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, macOS, or Linux

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/LatentPitch.git
cd LatentPitch
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Build Processed Datasets (Optional if Parquets are Included)
```bash
python data_loader.py
```

### 6. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
LatentPitch/
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # MIT License
├── README.md                       # Repository documentation
├── requirements.txt                # Production dependencies
├── app.py                          # Streamlit application entrypoint & UI
├── data_loader.py                  # ETL pipeline, normalization & parquet generation
├── pca_engine.py                   # Latent PCA reduction & cosine similarity engine
│
├── Data/                           # Raw telemetry and processed parquets
│   ├── PERFECT_players_data-2025_2026.csv   # Chuong Trinh enriched dataset (171 cols)
│   ├── players_data-2024_2025.csv           # Historical FBref dataset (267 cols)
│   ├── players_data-2025_2026.csv           # Standard FBref dataset (102 cols)
│   ├── players_data-2026_2027.csv           # Standard FBref dataset (102 cols)
│   ├── processed_players_dataset_b.parquet  # Primary active parquet
│   └── processed_players_dataset_a.parquet  # Legacy baseline parquet
│
├── benchmarks/                     # Audit, verification, and baseline scripts
│   ├── capture_baseline.py         # Baseline capture utility
│   ├── run_full_comparison.py      # Automated dataset A vs B benchmark
│   ├── verify_switching.py         # Cache isolation testing
│   └── check_b_players.py          # Data integrity diagnostic
│
└── before_after_comparison/        # Historical audit logs and similarity reports
    ├── dataset_comparison_report.json
    └── baseline_*.json
```

---

## 📊 Telemetry Attribution & Data Sources

| Source Identifier | Campaign | Author / Curator | Feature Count | Key Attributes |
| :--- | :---: | :--- | :---: | :--- |
| **Dataset B (Primary)** | 2025/2026 | **Chuong Trinh** (Kaggle) | **171** | Expected goals ($xG$), expected assists ($xAG$), duels won, ground/aerial duel win %, dribbles, recoveries, final-third passes, shot-prevention. |
| **Dataset A (Legacy)** | 2025/2026 | **Hubert Sidorowicz** (Kaggle) | **102** | Counting stats (goals, assists, cards, minutes). Lacks populated expected event metrics. |
| **2024/2025 Dataset** | 2024/2025 | **Hubert Sidorowicz** (Kaggle) | **267** | Full FBref seasonal metrics ($SCA, GCA, PrgP, PrgC$, defensive third pressures). |
| **2026/2027 Dataset** | 2026/2027 | **Hubert Sidorowicz** (Kaggle) | **102** | Standard forward-looking summary metrics. |

---

## 🔬 Methodology Summary

### Mathematical Pipeline
1. **Per-90 Scaling**: Eliminates playing time bias via $\text{Metric}_{\text{per90}} = (\text{Metric} / \text{Minutes}) \times 90$.
2. **Positional Imputation**: Missing values filled via cohort median $\hat{x}_{ij} = \text{median}(\text{Metric} \mid \text{Pos}, \text{Season})$.
3. **Cohort Isolation**: Goalkeepers and Outfield players operate in strictly isolated latent spaces.
4. **Z-Score Normalization**: $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$.
5. **Orthogonal Decomposition**: Covariance matrix $\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ decomposed into $k$ eigenvectors with sign-alignment.
6. **Cosine Similarity**: $\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \times 100\%$.

### Dimensionality Variance Trade-off
* **7 PCs**: 76.57% variance (ideal for minimal 7-axis radar charts).
* **10 PCs (Default)**: **83.66% variance** (optimal trade-off; captures wide crossing vs cutback creation, duel recovery intensity, and box penalty threat).
* **12 PCs**: 87.39% variance (high-spectrum tracking including aerial win rates and discipline balance).

---

## ⚠️ Analytical Limitations

* **Tactical System & Possession Bias**: Players in high-possession teams (e.g., Manchester City, Bayern Munich) naturally accumulate higher passing and touch volumes than players in low-block counter-attacking setups.
* **Unweighted Cross-League Strength**: Metrics are not adjusted by external league ELO coefficients.
* **On-Ball Event Truncation**: Operates on event telemetry; off-ball movement, pressing runs, and defensive shape maintenance are unmeasured without 25 Hz optical tracking.
* **Linearity**: PCA represents latent traits through linear combinations; non-linear interactions are not explicitly modeled.
* **Sample Cutoff**: Minimum 450 minutes threshold prevents small-sample noise but omits short-cameo prospects.

---

## 👥 Project Leadership & AI Collaboration Statement

* **Technical Director & Principal Architect**: Lead Developer / Researcher ([z3757181@gmail.com](mailto:z3757181@gmail.com))
  * **Domain & Architectural Direction**: Formulated the research hypothesis and requirements, designed the dual-engine latent architecture, diagnosed and proved the multi-season dimensional dilution failure mode, selected the 10-component variance optimization threshold based on tactical domain requirements, directed the role-conditioned preprocessing pipeline, and oversaw all empirical validation benchmarks.
* **AI Implementation Partner**: Antigravity (Google DeepMind Agentic Coding Framework)
  * **Computational Synthesis & Pair Programming**: Leveraged advanced agentic AI as an implementation and synthesis assistant under the Technical Director's prompt specifications, algorithmic guidance, and code-review oversight—generating boilerplate, refactoring pipelines, executing data transformations, and assisting with syntax diagnostics.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

