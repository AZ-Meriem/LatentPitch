import os
import json
import pandas as pd
import numpy as np
from pca_engine import PCAEngine

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'before_after_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Dataset A
path_a = os.path.join(DATA_DIR, 'processed_players_dataset_a.parquet')
df_a = pd.read_parquet(path_a)
engine_a = PCAEngine(n_components=7)
df_pca_a = engine_a.fit_transform_global(df_a)

# 2. Load Dataset B
path_b = os.path.join(DATA_DIR, 'processed_players_dataset_b.parquet')
df_b = pd.read_parquet(path_b)
engine_b = PCAEngine(n_components=7)
df_pca_b = engine_b.fit_transform_global(df_b)

season = '2025-2026'
min_minutes = 450
filtered_a = df_pca_a[(df_pca_a['Season'] == season) & (df_pca_a['Min'] >= min_minutes)].copy()
filtered_b = df_pca_b[(df_pca_b['Season'] == season) & (df_pca_b['Min'] >= min_minutes)].copy()

targets = [
    ('Rayan Cherki', 'MF', 'Playmaker / Attacking Midfielder'),
    ('Erling Haaland', 'FW', 'Elite Striker / Goalscorer'),
    ('Virgil van Dijk', 'DF', 'Ball-playing Center Back'),
    ('David Raya', 'GK', 'Sweeper Goalkeeper')
]

comparison_summary = {}

for target_name, pos, role_desc in targets:
    # A search
    match_a = filtered_a[filtered_a['Player'] == target_name]
    idx_a = match_a.index[0]
    row_a = filtered_a.loc[idx_a]
    top10_a = engine_a.get_similar_players(filtered_a, idx_a, top_n=10)
    
    # B search
    match_b = filtered_b[filtered_b['Player'] == target_name]
    idx_b = match_b.index[0]
    row_b = filtered_b.loc[idx_b]
    top10_b = engine_b.get_similar_players(filtered_b, idx_b, top_n=10)
    
    slug = target_name.lower().replace(' ', '_')
    
    # Save CSVs
    top10_a.to_csv(os.path.join(OUT_DIR, f'top10_dataset_a_{slug}.csv'), index=False)
    top10_b.to_csv(os.path.join(OUT_DIR, f'top10_dataset_b_{slug}.csv'), index=False)
    
    # Analyze changes
    players_a = top10_a['Player'].tolist()
    players_b = top10_b['Player'].tolist()
    common_matches = list(set(players_a).intersection(set(players_b)))
    
    comp_entry = {
        'player': target_name,
        'role': pos,
        'role_description': role_desc,
        'minutes_dataset_a': float(row_a['Min']),
        'minutes_dataset_b': float(row_b['Min']),
        'common_top10_matches': common_matches,
        'dataset_a_top10': top10_a[['Rank', 'Player', 'Squad', 'Comp', 'Age', 'Match %']].to_dict('records'),
        'dataset_b_top10': top10_b[['Rank', 'Player', 'Squad', 'Comp', 'Age', 'Match %']].to_dict('records'),
        'unique_to_dataset_a': [p for p in players_a if p not in players_b],
        'unique_to_dataset_b': [p for p in players_b if p not in players_a]
    }
    comparison_summary[target_name] = comp_entry

# Save comprehensive comparison json
with open(os.path.join(OUT_DIR, 'dataset_comparison_report.json'), 'w', encoding='utf-8') as f:
    json.dump(comparison_summary, f, indent=2, ensure_ascii=False)

print("Comparison run completed and saved to before_after_comparison/dataset_comparison_report.json")

