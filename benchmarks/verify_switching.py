import os
import pandas as pd
from pca_engine import PCAEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

def load_engine(source_key):
    if source_key == "Dataset B":
        data_path = os.path.join(DATA_DIR, 'processed_players_dataset_b.parquet')
    else:
        data_path = os.path.join(DATA_DIR, 'processed_players_dataset_a.parquet')
    df = pd.read_parquet(data_path)
    engine = PCAEngine(n_components=7)
    df_global = engine.fit_transform_global(df)
    return engine, df_global

print("1. Loading Dataset A...")
eng_a, df_a = load_engine("Dataset A")
filt_a = df_a[(df_a['Season'] == '2025-2026') & (df_a['Min'] >= 450)]
idx_cherki_a = filt_a[filt_a['Player'] == 'Rayan Cherki'].index[0]
res_cherki_a = eng_a.get_similar_players(filt_a, idx_cherki_a, top_n=5)
print("Top match A:", res_cherki_a['Player'].tolist())

print("\n2. Switching to Dataset B...")
eng_b, df_b = load_engine("Dataset B")
filt_b = df_b[(df_b['Season'] == '2025-2026') & (df_b['Min'] >= 450)]
idx_cherki_b = filt_b[filt_b['Player'] == 'Rayan Cherki'].index[0]
res_cherki_b = eng_b.get_similar_players(filt_b, idx_cherki_b, top_n=5)
print("Top match B:", res_cherki_b['Player'].tolist())

print("\n3. Switching back to Dataset A...")
eng_a2, df_a2 = load_engine("Dataset A")
filt_a2 = df_a2[(df_a2['Season'] == '2025-2026') & (df_a2['Min'] >= 450)]
idx_cherki_a2 = filt_a2[filt_a2['Player'] == 'Rayan Cherki'].index[0]
res_cherki_a2 = eng_a2.get_similar_players(filt_a2, idx_cherki_a2, top_n=5)
print("Top match A (re-check):", res_cherki_a2['Player'].tolist())

assert res_cherki_a['Player'].tolist() == res_cherki_a2['Player'].tolist(), "Error: Stale or mismatched results when switching back!"
print("\nVERIFICATION PASSED: No stale cache or cross-contamination between datasets.")

