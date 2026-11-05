import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pca_engine import PCAEngine
import os

st.set_page_config(
    layout="wide",
    page_title="Player Profiling Engine",
    initial_sidebar_state="expanded"
)

# Clean minimalist CSS
st.markdown("""
<style>
    body, .stApp { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3 { color: #e2e8f0; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #94a3b8; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #38bdf8; }
    .metric-bar { padding: 15px; background: #1e293b; border-left: 4px solid #38bdf8; display: flex; gap: 40px; margin-bottom: 25px; }
    .metric-col { display: flex; flex-direction: column; }
    .metric-col-title { font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; }
    .metric-col-value { font-size: 18px; font-weight: 600; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dataset_engine(source_key):
    if source_key == "Dataset B":
        data_path = os.path.join(os.path.dirname(__file__), 'Data', 'processed_players_dataset_b.parquet')
    else:
        data_path = os.path.join(os.path.dirname(__file__), 'Data', 'processed_players_dataset_a.parquet')
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(__file__), 'Data', 'processed_players.parquet')
            
    if not os.path.exists(data_path):
        st.error(f"Data missing at {data_path}. Run data_loader.py")
        st.stop()
    df = pd.read_parquet(data_path)
    engine = PCAEngine(n_components=12)
    df_global = engine.fit_transform_global(df)
    return engine, df_global

st.sidebar.title("Scouting Parameters")

# Advanced Engine Parameters (Expander)
with st.sidebar.expander("⚙️ Advanced Engine Configuration", expanded=False):
    st.caption("Underlying telemetry source, latent framework, and component depth.")
    dataset_options = [
        "Dataset B (PERFECT 2025/26 - Recommended)",
        "Dataset A (Standard FBref - Legacy)"
    ]
    selected_dataset_label = st.selectbox("Data Source", dataset_options, index=0)
    selected_dataset_key = "Dataset B" if "Dataset B" in selected_dataset_label else "Dataset A"

    calc_options = [
        "Season-Specific (Undiluted - Recommended)",
        "General Multi-Season (Global)"
    ]
    selected_calc = st.selectbox("Latent Framework", calc_options, index=0)
    pca_mode = 'season' if "Season-Specific" in selected_calc else 'global'
    pca_col_suffix = '_global' if pca_mode == 'global' else ''

    dim_options = [10, 7, 12]
    selected_dim = st.selectbox(
        "Latent Dimensions",
        dim_options,
        index=0,
        format_func=lambda x: f"{x} PCs" + (" (Recommended - 83.7% Var)" if x == 10 else " (Standard - 76.6% Var)" if x == 7 else " (Full Spectrum - 87.4% Var)")
    )

engine, df_global = load_dataset_engine(selected_dataset_key)

# Primary Filter Controls
season_opts = [s for s in ['2025-2026', 'All', '2024-2025', '2026-2027'] if s in df_global['Season'].unique()]
selected_season = st.sidebar.selectbox("Season Snapshot", season_opts, index=0)

all_leagues = ['All'] + sorted([c for c in df_global['Comp'].dropna().unique() if c.strip() != ''])
selected_league = st.sidebar.selectbox("League / Competition", all_leagues)

pos_options = ['All', 'FW', 'MF', 'DF', 'GK']
selected_pos = st.sidebar.selectbox("Positional Cohort", pos_options)

min_limit = int(df_global['Min'].min()) if not df_global['Min'].empty else 450
max_limit = int(df_global['Min'].max()) if not df_global['Min'].empty else 3800
selected_min_minutes = st.sidebar.slider("Minimum Minutes", min_limit, max_limit, 450, step=50)

min_age = int(df_global['Age'].min()) if not df_global['Age'].isna().all() else 15
max_age = int(df_global['Age'].max()) if not df_global['Age'].isna().all() else 45
selected_age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

filtered_df = df_global.copy()

if selected_season != 'All':
    filtered_df = filtered_df[filtered_df['Season'] == selected_season]
else:
    filtered_df = filtered_df[filtered_df['Season'] == 'All']

filtered_df = filtered_df[filtered_df['Min'] >= selected_min_minutes]
if selected_age_range != (min_age, max_age):
    filtered_df = filtered_df[(filtered_df['Age'] >= selected_age_range[0]) & (filtered_df['Age'] <= selected_age_range[1])]

if selected_league != 'All':
    filtered_df = filtered_df[filtered_df['Comp'] == selected_league]

if selected_pos != 'All':
    filtered_df = filtered_df[filtered_df['Pos'].astype(str).apply(lambda p: selected_pos in [x.strip() for x in p.split(',')])]

if filtered_df.empty:
    st.warning("No players found matching filter criteria.")
    st.stop()

available_players = sorted(filtered_df['Player'].unique())

if 'selected_player' not in st.session_state:
    st.session_state.selected_player = available_players[0]

if st.session_state.selected_player not in available_players:
    st.session_state.selected_player = available_players[0]

if 'target_player_select' in st.session_state and st.session_state.target_player_select not in available_players:
    st.session_state.target_player_select = st.session_state.selected_player

current_index = available_players.index(st.session_state.selected_player)

def on_player_change():
    st.session_state.selected_player = st.session_state.target_player_select

target_player = st.sidebar.selectbox(
    "Target Player Selection",
    options=available_players,
    index=current_index,
    key="target_player_select",
    on_change=on_player_change
)

target_matches = filtered_df[filtered_df['Player'] == target_player]
target_idx = target_matches.index[0]
target_row = filtered_df.loc[target_idx]
cohort = 'GK' if target_row['Primary_Pos'] == 'GK' else 'Outfield'

# Determine key metrics tailored by primary position and playing role
pri_pos = target_row['Primary_Pos']
pos_raw = str(target_row['Pos'])

if pri_pos == 'FW' or 'FW' in pos_raw:
    # Forward & Winger metrics (Yamal, Saka, Doku, Vinicius, Mbappe, Haaland, etc.)
    metrics = ['xG', 'npxG', 'Gls', 'Sh', 'SoT', 'xAG', 'KP', 'Dribbles', 'Touches']
elif pri_pos == 'MF':
    # Central/defensive midfielder metrics
    metrics = ['Cmp', 'Att', 'KP', 'xAG', 'Touches', 'FinalThirdPasses', 'Tkl', 'Int', 'Recov', 'DuelsWon']
elif pri_pos == 'DF':
    # Defender metrics
    metrics = ['TklW', 'Int', 'Clr', 'Blocks', 'DuelsWon', 'Cmp', 'Touches', 'Recov']
else:
    # Goalkeeper metrics
    metrics = ['Saves', 'CS', 'GA', 'Touches', 'Cmp']
    
pct_cols = [m + '_pct' for m in metrics if m + '_pct' in filtered_df.columns]
labels = [m.replace('_pct', '') for m in pct_cols]

st.title("Player Similarity & Profiling Engine")

st.markdown(f"""
<div class="metric-bar">
    <div class="metric-col"><span class="metric-col-title">Target Profile</span><span class="metric-col-value">{target_player}</span></div>
    <div class="metric-col"><span class="metric-col-title">Squad</span><span class="metric-col-value">{target_row['Squad']}</span></div>
    <div class="metric-col"><span class="metric-col-title">Competition</span><span class="metric-col-value">{target_row['Comp']}</span></div>
    <div class="metric-col"><span class="metric-col-title">Position</span><span class="metric-col-value">{target_row['Pos']}</span></div>
    <div class="metric-col"><span class="metric-col-title">Age</span><span class="metric-col-value">{int(target_row['Age']) if pd.notna(target_row['Age']) else 'N/A'}</span></div>
    <div class="metric-col"><span class="metric-col-title">Minutes</span><span class="metric-col-value">{int(target_row['Min']):,}</span></div>
</div>
""", unsafe_allow_html=True)

similar_players_df = engine.get_similar_players(filtered_df, target_idx, top_n=10, mode=pca_mode, n_components=selected_dim)

tab_scout, tab_visuals, tab_manuscript = st.tabs(["Similarity & Projection", "Advanced Scouting Visuals", "Research Manuscript & Methodology"])

with tab_scout:
    st.subheader(f"Top Latent Matches ({selected_dim} PCs)")
    
    st.dataframe(
        similar_players_df[['Rank', 'Player', 'Squad', 'Comp', 'Age', 'Match %']],
        hide_index=True,
        use_container_width=True
    )
    
    with st.expander(f"Compare 7 vs 10 vs 12 PC Dimensionality for {target_player}"):
        sim_7 = engine.get_similar_players(filtered_df, target_idx, top_n=10, mode=pca_mode, n_components=7)
        sim_10 = engine.get_similar_players(filtered_df, target_idx, top_n=10, mode=pca_mode, n_components=10)
        sim_12 = engine.get_similar_players(filtered_df, target_idx, top_n=10, mode=pca_mode, n_components=12)
        
        comp_data = []
        for r in range(10):
            r7 = sim_7.iloc[r] if r < len(sim_7) else None
            r10 = sim_10.iloc[r] if r < len(sim_10) else None
            r12 = sim_12.iloc[r] if r < len(sim_12) else None
            comp_data.append({
                'Rank': r + 1,
                '7 PCs Match': f"{r7['Player']} ({r7['Match %']})" if r7 is not None else "-",
                '10 PCs Match': f"{r10['Player']} ({r10['Match %']})" if r10 is not None else "-",
                '12 PCs Match': f"{r12['Player']} ({r12['Match %']})" if r12 is not None else "-"
            })
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Dual-Player Direct Comparison")
    
    match_opts = [f"{r['Rank']}. {r['Player']} ({r['Squad']})" for _, r in similar_players_df.iterrows()]
    selected_match = st.selectbox("Select Matched Player to Compare Against Target Player", match_opts, index=0)
    compare_idx = similar_players_df[similar_players_df['Rank'] == int(selected_match.split('.')[0])]['Index'].iloc[0]
    compare_row = filtered_df.loc[compare_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pct_cols:
            target_pcts = target_row[pct_cols].values.tolist()
            compare_pcts = compare_row[pct_cols].values.tolist()
            
            # Close the radar loop
            target_pcts += [target_pcts[0]]
            compare_pcts += [compare_pcts[0]]
            labels_loop = labels + [labels[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=target_pcts,
                theta=labels_loop,
                fill='toself',
                name=f"{target_player}",
                line_color='#38bdf8'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=compare_pcts,
                theta=labels_loop,
                fill='toself',
                name=f"{compare_row['Player']}",
                line_color='#ef4444'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Positional Percentiles (per 90)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
    with col2:
        dyn_labels_dict = engine.get_dynamic_labels(cohort, season=selected_season, mode=pca_mode, n_components=selected_dim)
        pca_cols = [f'PC{i+1}{pca_col_suffix}' for i in range(selected_dim)]
        
        target_pca_mm = target_row[[f'{c}_mm' for c in pca_cols]].values.tolist()
        compare_pca_mm = compare_row[[f'{c}_mm' for c in pca_cols]].values.tolist()
        
        pca_labels = [dyn_labels_dict.get(f'PC{i+1}', f'PC{i+1}') for i in range(selected_dim)]
        
        # Close the radar loop
        target_pca_mm += [target_pca_mm[0]]
        compare_pca_mm += [compare_pca_mm[0]]
        pca_labels_loop = pca_labels + [pca_labels[0]]
        
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatterpolar(
            r=target_pca_mm,
            theta=pca_labels_loop,
            fill='toself',
            name=f"{target_player}",
            line_color='#38bdf8'
        ))
        fig_pca.add_trace(go.Scatterpolar(
            r=compare_pca_mm,
            theta=pca_labels_loop,
            fill='toself',
            name=f"{compare_row['Player']}",
            line_color='#ef4444'
        ))
        fig_pca.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True,
            title=f"{selected_dim}D Latent Component Overlay",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.subheader("3D PCA Vector Projection")
    df_plot = filtered_df.copy()
    df_plot['Status'] = 'Cohort Player'
    df_plot.loc[similar_players_df['Index'], 'Status'] = 'Top Match'
    df_plot.loc[target_idx, 'Status'] = 'Target Player'
    
    fig_3d = px.scatter_3d(
        df_plot, x=pca_cols[0], y=pca_cols[1], z=pca_cols[2],
        color='Status', hover_name='Player', hover_data=['Squad', 'Comp', 'Pos', 'Min'],
        color_discrete_map={'Target Player': '#38bdf8', 'Top Match': '#ef4444', 'Cohort Player': '#334155'},
        template="plotly_dark"
    )
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab_visuals:
    st.subheader("Metric Correlation Quadrant Plot")
    
    per90_cols = [c for c in filtered_df.columns if c.endswith('_per90')]
    
    col1, col2 = st.columns(2)
    with col1:
        x_default = per90_cols.index('PrgC_per90') if 'PrgC_per90' in per90_cols else 0
        x_metric = st.selectbox("X-Axis Metric", per90_cols, index=x_default)
    with col2:
        y_default = per90_cols.index('xAG_per90') if 'xAG_per90' in per90_cols else 1
        y_metric = st.selectbox("Y-Axis Metric", per90_cols, index=y_default)
        
    df_plot_quad = filtered_df.copy()
    df_plot_quad['Plot_Status'] = 'Cohort Player'
    df_plot_quad['Plot_Size'] = 10
    
    top5_matches = similar_players_df.head(5)['Index'].tolist()
    df_plot_quad.loc[top5_matches, 'Plot_Status'] = 'Top 5 Match'
    df_plot_quad.loc[target_idx, 'Plot_Status'] = 'Target Player'
    df_plot_quad.loc[top5_matches, 'Plot_Size'] = 15
    df_plot_quad.loc[target_idx, 'Plot_Size'] = 20
    
    df_plot_quad = pd.concat([
        df_plot_quad[df_plot_quad['Plot_Status'] == 'Cohort Player'],
        df_plot_quad[df_plot_quad['Plot_Status'] == 'Top 5 Match'],
        df_plot_quad[df_plot_quad['Plot_Status'] == 'Target Player']
    ])
    
    fig_quad = px.scatter(
        df_plot_quad, x=x_metric, y=y_metric, 
        color='Plot_Status', 
        hover_name='Player', hover_data=['Squad', 'Comp', 'Pos', 'Age'],
        color_discrete_map={'Target Player': '#06b6d4', 'Top 5 Match': '#f97316', 'Cohort Player': 'rgba(148, 163, 184, 0.3)'},
        size='Plot_Size',
        size_max=15,
        text=df_plot_quad.apply(lambda r: r['Player'] if r['Plot_Status'] != 'Cohort Player' else '', axis=1),
        template="plotly_dark"
    )
    fig_quad.update_traces(textposition='top center')
    
    x_median = filtered_df[x_metric].median()
    y_median = filtered_df[y_metric].median()
    fig_quad.add_hline(y=y_median, line_dash="dot", line_color="gray", opacity=0.5)
    fig_quad.add_vline(x=x_median, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig_quad.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_quad, use_container_width=True)

    st.markdown("---")
    st.subheader("Percentile Delta Diverging Bar Chart")
    
    if pct_cols:
        delta_pcts = target_row[pct_cols].values - compare_row[pct_cols].values
        delta_df = pd.DataFrame({
            'Metric': labels,
            'Delta': delta_pcts
        }).sort_values('Delta')
        
        fig_delta = px.bar(
            delta_df, x='Delta', y='Metric', orientation='h',
            color='Delta', color_continuous_scale=['#f97316', '#334155', '#06b6d4'],
            color_continuous_midpoint=0,
            text=delta_df['Delta'].round(1).astype(str),
            template="plotly_dark",
            title=f"{target_player} vs. {compare_row['Player']} (Target - Match)"
        )
        fig_delta.update_traces(textposition='outside')
        fig_delta.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.info("No percentiles found for this cohort.")
    
    st.markdown("---")
    st.subheader("Top 10 Matches Heatmap Matrix")
    
    if pct_cols:
        top10_idx = similar_players_df['Index'].tolist()
        matrix_indices = [target_idx] + top10_idx
        matrix_df = filtered_df.loc[matrix_indices].copy()
        
        matrix_cols = pct_cols[:6]
        disp_df = matrix_df[['Player', 'Squad'] + matrix_cols].copy()
        disp_df.columns = ['Player', 'Squad'] + [c.replace('_pct', '') for c in matrix_cols]
        
        st.dataframe(
            disp_df.style.background_gradient(cmap='viridis', subset=disp_df.columns[2:]),
            hide_index=True,
            use_container_width=True
        )

with tab_manuscript:
    st.subheader(f"Latent Dimensional Diagnostics ({selected_dim} PCs Active)")
    loadings = engine.get_loadings(cohort, season=selected_season, mode=pca_mode, n_components=selected_dim)
    dyn_labels_dict = engine.get_dynamic_labels(cohort, season=selected_season, mode=pca_mode, n_components=selected_dim)
    pca_cols = [f'PC{i+1}{pca_col_suffix}' for i in range(selected_dim)]
    
    vec_data = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(selected_dim)],
        'Dynamic Label': [dyn_labels_dict.get(f'PC{i+1}', '') for i in range(selected_dim)],
        'Vector Score (Z)': target_row[pca_cols].values
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(vec_data, hide_index=True, use_container_width=True)
    with col2:
        if not loadings.empty:
            top_features = []
            for pc in loadings.columns:
                top_idx = loadings[pc].abs().idxmax()
                top_features.append(f"{top_idx} ({loadings.loc[top_idx, pc]:.2f})")
            desc_df = pd.DataFrame({
                'Component': loadings.columns,
                'Dynamic Label': [dyn_labels_dict.get(c, '') for c in loadings.columns],
                'Dominant Feature & Loading': top_features
            })
            st.dataframe(desc_df, hide_index=True, use_container_width=True)
            
    st.markdown(r"""
    ---
    # LatentPitch: High-Dimensional Latent Profiling & Scouting Architecture
    **Technical Research Manuscript & Methodological Documentation**
    
    ### Abstract
    Traditional football player scouting relies heavily on either isolated counting totals (which suffer from severe playing-time bias) or high-dimensional scatter searches that fall prey to multicollinearity and the curse of dimensionality. **LatentPitch** implements a dual-architecture latent profiling engine that standardizes multi-league telemetry across the top European competitions, performs cohort-conditioned orthogonal decomposition, and projects players into a calibrated latent vector space ($k \in \{7, 10, 12\}$). This document details the mathematical formulations, telemetry sources, exploratory options evaluated and discarded, and fundamental analytical limitations of the system.

    ---

    ### Project Leadership & AI Collaboration Statement
    * **Technical Director & Principal Architect**: Lead Developer / Researcher ([z3757181@gmail.com](mailto:z3757181@gmail.com))
      * *Core Contribution*: Formulated the core research hypothesis and analytical requirements, designed the dual-engine latent architecture, discovered and mathematically diagnosed the multi-season zero-fill dilution failure mode, determined the 10-component variance optimization threshold based on tactical domain requirements, specified the role-conditioned preprocessing pipeline, and directed all empirical validation benchmarks.
    * **AI Implementation Partner**: Antigravity (Google DeepMind Agentic Coding Framework)
      * *Core Contribution*: Operated as an AI pair-programming assistant under the Technical Director's prompt specifications, algorithmic guidance, and code-review oversight—generating implementation code, pipeline refactoring, data transformation routines, and syntax diagnostics.

    ---
    
    ### 1. Data Attribution & Multi-Source Telemetry Inventory
    
    LatentPitch integrates multiple independent data sources across European top-flight football (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) across three seasons:
    
    * **Dataset B (2025/2026 Season - Primary Enriched Baseline)**:
      * **Author / Attribution**: Sourced from **Chuong Trinh on Kaggle** (`PERFECT_players_data-2025_2026.csv`).
      * **Parameter Count**: **171 raw parameters**.
      * **Characteristics**: The most detailed event-level telemetry in the platform. Features include playing time distributions, expected goals ($xG$), expected assists ($xAG$), key pass rates, touch heat volumes, successful dribbles, ground and aerial duel win percentages, ball recovery zones, progressive deliveries, final-third passes, and goalkeeper shot-prevention metrics.
      * **Role in App**: Configured as the **Default Engine Source** for the 2025/2026 campaign.
    * **Dataset A (2025/2026 Season - Standard FBref Baseline)**:
      * **Author / Attribution**: Sourced from **Hubert Sidorowicz on Kaggle** (`players_data-2025_2026.csv`).
      * **Parameter Count**: **102 parameters**.
      * **Characteristics**: Basic summary counts (goals, assists, cards, minutes, standard passes). Lacks granular expected event telemetry and progressive pass/carry distance tracking. Maintained in the Advanced Configuration for legacy baseline auditing.
    * **2024/2025 Season Dataset**:
      * **Author / Attribution**: Compiled by **Hubert Sidorowicz on Kaggle** (`players_data-2024_2025.csv`).
      * **Parameter Count**: **267 parameters**.
      * **Characteristics**: Complete historical FBref seasonal metrics with fully populated shot-creating actions ($SCA$), goal-creating actions ($GCA$), progressive passes ($PrgP$), progressive carries ($PrgC$), and defensive third pressure metrics.
    * **2026/2027 Season Dataset**:
      * **Author / Attribution**: Compiled by **Hubert Sidorowicz on Kaggle** (`players_data-2026_2027.csv`).
      * **Parameter Count**: **102 parameters** with standard summary metrics.

    ---

    ### 2. Mathematical Framework & Preprocessing Pipeline
    
    ```
    Raw Event Data  -->  Per-90 Normalization  -->  Cohort Imputation  -->  Z-Score Scaling  -->  Sign-Aligned PCA  -->  Cosine Similarity
    (170+ Features)      (Volume Independence)       (Positional Medians)    (Unit Variance)       (Orthogonal Latent)      (Angular Match %)
    ```

    1. **Per-90 Volumetric Normalization**:
       To prevent substitution and playing-time bias, all counting metrics are normalized relative to 90-minute equivalents:
       $$\text{Metric}_{\text{per90}} = \frac{\text{Metric}}{90s} = \frac{\text{Metric} \times 90}{\text{Minutes}}$$
    2. **Sample Filtration**:
       A strict sample filter ($Min \ge 450\text{ minutes}$) eliminates low-sample noise and volatile short-stint outliers.
    3. **Positional Cohort Imputation**:
       Missing feature values are imputed using the positional median calculated strictly within each primary role cohort ($\text{Primary\_Pos} \in \{\text{FW}, \text{MF}, \text{DF}, \text{GK}\}$):
       $$\hat{x}_{ij} = \text{median}\left(\{x_{kj} \mid \text{Pos}_k = \text{Pos}_i, \text{Season}_k = \text{Season}_i\}\right)$$
    4. **Positional Percentiles**:
       Percentiles are pre-computed strictly relative to primary role and active season:
       $$\text{Pct}_{ij} = \text{Rank}_{\%}(x_{ij} \mid \text{Pos}_i, \text{Season}_i) \times 100$$
    5. **Orthogonal Latent Decomposition (PCA)**:
       Standardized feature matrices $\mathbf{Z} \in \mathbb{R}^{N \times P}$ are projected onto orthogonal eigenvectors:
       $$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{Z}^T \mathbf{Z} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T, \quad \mathbf{Y}_k = \mathbf{Z} \mathbf{V}_k$$
       *Sign-alignment constraint*: For each principal component $j$, the eigenvector sign is forced positive relative to the feature with maximum absolute loading: $\mathbf{v}_j \leftarrow -\mathbf{v}_j$ if $v_{j, \arg\max |v_{j}|} < 0$.
    6. **Latent Metric Space (Cosine Similarity)**:
       Target similarity is measured via unweighted cosine similarity across the active $k$-dimensional latent space ($k=10$ by default):
       $$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^k u_i v_i}{\sqrt{\sum_{i=1}^k u_i^2} \sqrt{\sum_{i=1}^k v_i^2}}$$

    ---

    ### 3. Evolutionary Journey: Options Explored, Discarded, & Adopted
    
    #### A. Exploration 1: Global Multi-Season PCA Pooling vs. Dilution
    * **Hypothesis**: Fitting a single global PCA model across all seasons simultaneously would create a unified cross-era coordinate system.
    * **Failure Mode (Discarded as Default)**: Because the 2024/25 dataset contained 267 features while 2025/26 and 2026/27 contained 102–171 features, missing columns were zero-filled across 50%+ of the dataset. This caused:
      1. Artificial suppression of feature variance in 2024/25.
      2. **Dimensional Collapse in 2025/26**: PC2 loaded heavily on 2024/25-only metrics ($SCA, PPA, PrgP$), collapsing PC2 near zero for all 2025/26 players. Attacking players were compressed onto a single diagonal driven only by PC1 and PC3.
      3. **False Equivalence**: Marcus Tavernier ranked #2 and deep midfielder Enzo Fernández ranked #16 for Lamine Yamal, while Michael Olise artificially registered a 97.44% match (#3).
    * **Adopted Solution**: Implemented the **Season-Specific (Undiluted)** architecture as the platform default, fitting PCA strictly on active non-zero variance. The Global framework is retained in Advanced Configuration for macro multi-season evaluation.

    #### B. Exploration 2: Latent Dimensionality Spectrum ($k \in \{7, 10, 12\}$)
    * **Evaluated Choices**:
      * **7 PCs (Standard)**: Captured **76.57%** of active variance. Effective for clean polar radar visualization, but discarded nuances such as crossing vs cutbacks and recovery duel intensity.
      * **10 PCs (Adopted as Optimal Default)**: Captures **83.66%** of active variance. Unpacks essential secondary tactical dimensions (crossing volume, set-piece delivery, non-penalty xG threat, recovery pressing) without inducing high-dimensional hubness or distance concentration.
      * **12 PCs (Full Spectrum)**: Captures **87.39%** of active variance. Further incorporates aerial contest ratios and discipline balance.
    * **Tactical Impact on Yamal vs Olise**:
      * In 7 PCs, Yamal and Olise diverge moderately ($\Delta = 12.6\%$).
      * In 10 PCs, the model clearly isolates Yamal as an elite 1v1 take-on inside forward (Musiala #1 at 96.8%, Kvaratskhelia #2 at 96.4%, Doué #3 at 93.0%, Saka #4 at 92.2%), correctly recognizing Olise's distinct profile as a wide set-piece crosser (5.68 crosses/90 vs Yamal's 3.07, and 2.53 dribbles vs Yamal's 5.30).

    #### C. Engineering Fixes & Bug Resolutions
    * **Age Parsing Bug in Bundesliga & Ligue 1**: In the raw Chuong Trinh dataset, ages for German and French leagues were recorded with hyphenated birth offsets (e.g., `'21-066'`), while other leagues used integer strings (`'27'`). A pandas `dtype == object` check failed under modern pandas `pd.StringDtype`, causing direct coercion to `NaN`. Because the sidebar age slider filtered `(Age >= min) & (Age <= max)`, **100% of Bundesliga and Ligue 1 players were silently eliminated**. Fixed via unconditional regex hyphen-splitting and defensive missing-age handling.
    * **Winger Positional Misclassification**: Dual-position players tagged `MF,FW` (including Lamine Yamal) were previously string-split to `MF`, causing forwards to receive central/defensive midfielder radar percentiles. Resolved by updating `assign_primary_pos()` to categorize any player with `FW` anywhere in their role string as Forward/Winger.

    ---

    ### 4. Methodological & Analytical Limitations
    
    While LatentPitch provides robust mathematical profiling, practitioners should interpret similarities within the context of the following fundamental limitations:
    
    1. **Tactical System & Possession Bias**:
       Event volumes are inherently influenced by team tactical game-plans. Players in dominant, high-possession sides (e.g., Manchester City, Bayern Munich, Barcelona) accumulate substantially higher pass counts, touches, and progressive receptions than identically skilled players operating in low-block, transition-based counter-attacking systems.
    2. **Unweighted Cross-League Strength Differentials**:
       Similarity metrics treat events across the Premier League, La Liga, Serie A, Bundesliga, and Ligue 1 equally. A player completing 4.0 dribbles per 90 in one league is not mathematically adjusted for opponent defensive resistance without external league-coefficient weighting (e.g., UEFA ELO coefficients).
    3. **Absence of Spatio-Temporal Tracking Telemetry**:
       The platform operates on event-level data (on-ball actions: shots, passes, duels). It does **not** ingest continuous optical tracking data (25 Hz coordinate streams). Consequently, critical off-ball aspects—such as pressing run velocity, defensive shape recovery, decoy runs, and passing lane occlusion—are unmeasured.
    4. **Linear Orthogonality Assumption**:
       PCA assumes that latent archetypes can be represented via linear combinations of observed metrics. While effective and transparent, it cannot capture complex non-linear manifolds or multi-modal skill topologies that non-linear deep learning representations (e.g., Variational Autoencoders) could theoretically model.
    5. **Sample Truncation Thresholds**:
       The 450-minute threshold prevents noise from volatile 1-game cameos, but inherently excludes emerging academy debuts or winter transfer window arrivals who have logged under 5 full matches.
    """)
