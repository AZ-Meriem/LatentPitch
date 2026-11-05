import os
import json
import pandas as pd
import numpy as np
from pca_engine import PCAEngine

# Load current processed parquet
data_path = os.path.join(os.path.dirname(__file__), 'Data', 'processed_players.parquet')
df = pd.read_parquet(data_path)

engine = PCAEngine(n_components=7)
df_global = engine.fit_transform_global(df)

season = '2025-2026'
min_minutes = 450
targets = ['Rayan Cherki', 'Erling Haaland', 'Virgil van Dijk', 'David Raya']

filtered_df = df_global[(df_global['Season'] == season) & (df_global['Min'] >= min_minutes)].copy()

out_dir = os.path.join(os.path.dirname(__file__), 'before_after_comparison')
os.makedirs(out_dir, exist_ok=True)

pca_cols = [f'PC{i+1}' for i in range(7)]
per90_cols = [c for c in df_global.columns if c.endswith('_per90')]
pct_cols = [c for c in df_global.columns if c.endswith('_pct')]

for target_player_name in targets:
    matches = filtered_df[filtered_df['Player'] == target_player_name]
    if matches.empty:
        print(f"Player {target_player_name} not found in {season}")
        continue

    target_idx = matches.index[0]
    target_row = filtered_df.loc[target_idx]
    cohort = 'GK' if target_row['Primary_Pos'] == 'GK' else 'Outfield'
    dyn_labels = engine.get_dynamic_labels(cohort)

    similar_df = engine.get_similar_players(filtered_df, target_idx, top_n=10)

    slug = target_player_name.lower().replace(' ', '_')
    csv_file = os.path.join(out_dir, f'baseline_{slug}_top10.csv')
    similar_df.to_csv(csv_file, index=False, encoding='utf-8')

    profile = {
        'metadata': {
            'player': target_player_name,
            'squad': target_row['Squad'],
            'comp': target_row['Comp'],
            'pos': target_row['Pos'],
            'role': target_row['Primary_Pos'],
            'season': season,
            'min_minutes': min_minutes,
            'dataset': 'Dataset A (player_data_2025-2026)'
        },
        'pca_vectors': {col: float(target_row[col]) for col in pca_cols},
        'dynamic_labels': dyn_labels,
        'per90_stats': {col: float(target_row[col]) for col in per90_cols if pd.notna(target_row[col])},
        'percentiles': {col: float(target_row[col]) for col in pct_cols if pd.notna(target_row[col])},
        'top_10_similar': similar_df.to_dict(orient='records')
    }

    json_file = os.path.join(out_dir, f'baseline_{slug}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"Baseline saved for {target_player_name} ({target_row['Primary_Pos']})")
