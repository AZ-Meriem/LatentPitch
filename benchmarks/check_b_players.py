import pandas as pd
from pca_engine import PCAEngine

df_b = pd.read_parquet('Data/processed_players_dataset_b.parquet')
engine = PCAEngine(n_components=7)
df_pca = engine.fit_transform_global(df_b)

filt_b = df_pca[(df_pca['Season'] == '2025-2026') & (df_pca['Min'] >= 450)].copy()

print('Total 2025-2026 players in B with Min>=450:', len(filt_b))
print('By Comp:')
print(filt_b['Comp'].value_counts())

for target in ['Harry Kane', 'Florian Wirtz', 'Bradley Barcola', 'Erling Haaland', 'Lamine Yamal']:
    m = filt_b[filt_b['Player'] == target]
    if m.empty:
        print(f'{target}: NOT FOUND!')
    else:
        idx = m.index[0]
        sim = engine.get_similar_players(filt_b, idx, top_n=10, mode='season')
        print(f"\nTop 10 for {target} ({m['Comp'].iloc[0]}):")
        for _, r in sim.iterrows():
            print(f"  {r['Rank']}. {r['Player']} ({r['Squad']}, {r['Comp']}) - {r['Match %']}")

