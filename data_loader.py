import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

def clean_comp(comp_str):
    comp_str = str(comp_str).strip()
    c_lower = comp_str.lower()
    if 'bundesliga' in c_lower:
        return 'Bundesliga'
    elif 'ligue 1' in c_lower:
        return 'Ligue 1'
    elif 'premier league' in c_lower:
        return 'Premier League'
    elif 'la liga' in c_lower:
        return 'La Liga'
    elif 'serie a' in c_lower:
        return 'Serie A'
    return comp_str

def assign_primary_pos(pos_str):
    p = str(pos_str).strip()
    if 'GK' in p:
        return 'GK'
    # Any player with FW role (pure FW, MF,FW wingers, FW,MF attackers) gets categorized as FW
    elif 'FW' in p:
        return 'FW'
    elif 'DF' in p and not p.startswith('MF'):
        return 'DF'
    return 'MF'

# Standard counting statistics across the ecosystem
COUNTING_STATS = [
    'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'Sh', 'SoT', 'Crs',
    'TklW', 'Int', 'Fls', 'Fld', 'Off', 'OG', 'CrdY', 'CrdR',
    'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR',
    'Cmp', 'Att', 'TotDist', 'PrgDist', 'KP', '1/3', 'PPA', 'CrsPA',
    'SCA', 'GCA', 'Tkl', 'Blocks', 'Clr', 'Touches', 'Carries', 'Rec', 'Recov',
    'GA', 'SoTA', 'Saves', 'CS',
    'Dribbles', 'DuelsWon', 'FinalThirdPasses', 'BigChancesCreated'
]

def load_and_standardize_2025_b():
    """Load and standardize Chuong Trinh's PERFECT 2025/2026 dataset (171 columns)."""
    b_file = os.path.join(DATA_DIR, 'PERFECT_players_data-2025_2026.csv')
    df_raw = pd.read_csv(b_file)
    
    rename_map = {
        'player': 'Player',
        'nation_': 'Nation',
        'pos_': 'Pos',
        'team': 'Squad',
        'age_': 'Age',
        'Playing Time_Min': 'Min',
        'Playing Time_90s': '90s',
        'Playing Time_MP': 'MP',
        'Playing Time_Starts': 'Starts',
    }
    df_b = df_raw.rename(columns=rename_map).copy()
    df_b['Comp'] = df_b['league'].apply(clean_comp)
    df_b['Season'] = '2025-2026'

    for col in ['Min', '90s', 'MP', 'Starts', 'Age']:
        if col in df_b.columns:
            df_b[col] = pd.to_numeric(df_b[col].astype(str).str.replace(',', '', regex=False).str.split('-').str[0], errors='coerce')

    stat_mapping = {
        'Gls': 'Performance_Gls',
        'Ast': 'Performance_Ast',
        'G+A': 'Performance_G+A',
        'G-PK': 'Performance_G-PK',
        'PK': 'Performance_PK',
        'PKatt': 'Performance_PKatt',
        'Sh': 'Standard_Sh',
        'SoT': 'Standard_SoT',
        'Crs': 'Performance_Crs',
        'TklW': 'tacklesWon',
        'Int': 'interceptions',
        'Fls': 'Performance_Fls',
        'Fld': 'Performance_Fld',
        'Off': 'Performance_Off',
        'OG': 'Performance_OG',
        'CrdY': 'Performance_CrdY',
        'CrdR': 'Performance_CrdR',
        'xG': 'expectedGoals',
        'npxG': 'np_xg_under',
        'xAG': 'expectedAssists',
        'KP': 'keyPasses',
        'Tkl': 'tackles',
        'Blocks': 'blockedShots',
        'Clr': 'clearances',
        'Touches': 'touches',
        'Recov': 'ballRecovery',
        'Cmp': 'accuratePasses',
        'Att': 'totalPasses',
        'GA': 'goalsConceded',
        'Saves': 'saves',
        'CS': 'cleanSheet',
        'Dribbles': 'successfulDribbles',
        'DuelsWon': 'totalDuelsWon',
        'FinalThirdPasses': 'accurateFinalThirdPasses',
        'BigChancesCreated': 'bigChancesCreated'
    }

    for target_col, src_col in stat_mapping.items():
        if src_col in df_b.columns:
            df_b[target_col] = pd.to_numeric(df_b[src_col], errors='coerce').fillna(0)
        else:
            df_b[target_col] = 0.0

    return df_b


def load_standard_season_file(filename, season_label):
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    df['Season'] = season_label
    df['Comp'] = df['Comp'].apply(clean_comp)
    
    for col in ['Min', '90s', 'MP', 'Starts', 'Age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False).str.split('-').str[0], errors='coerce')
                
    for col in COUNTING_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
            
    return df


def process_pipeline(season_dfs, output_filename):
    """Generic processing pipeline that creates single-season, multi-season ('All'), and percentiles."""
    df_raw = pd.concat(season_dfs, ignore_index=True)
    
    # Clean positions
    df_raw['Pos'] = df_raw['Pos'].fillna('MF').astype(str)
    df_raw['Primary_Pos'] = df_raw['Pos'].apply(assign_primary_pos)

    # 1. Single season records (Min >= 450)
    df_single = df_raw[df_raw['Min'] >= 450].copy()
    for col in COUNTING_STATS:
        df_single[col + '_per90'] = np.where(df_single['90s'] > 0, df_single[col] / df_single['90s'], 0)
        
    # 2. Multi-Season Aggregated Records ('All')
    agg_dict = {
        'Min': 'sum',
        '90s': 'sum',
        'MP': 'sum',
        'Starts': 'sum',
        'Pos': lambda s: s.mode().iloc[0] if not s.empty else s.iloc[0],
        'Primary_Pos': lambda s: s.mode().iloc[0] if not s.empty else s.iloc[0],
        'Squad': lambda s: s.iloc[-1] if not s.empty else '',
        'Comp': lambda s: s.iloc[-1] if not s.empty else '',
        'Age': 'max',
        'Nation': lambda s: s.iloc[-1] if not s.empty else ''
    }
    for col in COUNTING_STATS:
        agg_dict[col] = 'sum'
        
    df_multi = df_raw.groupby('Player', as_index=False).agg(agg_dict)
    df_multi['Season'] = 'All'
    df_multi = df_multi[df_multi['Min'] >= 450].copy()
    for col in COUNTING_STATS:
        df_multi[col + '_per90'] = np.where(df_multi['90s'] > 0, df_multi[col] / df_multi['90s'], 0)
        
    # 3. Combine Single and Multi
    df_combined = pd.concat([df_single, df_multi], ignore_index=True)
    
    # 4. Impute positional medians strictly within cohort
    per90_features = [c + '_per90' for c in COUNTING_STATS]
    for col in per90_features:
        pos_median = df_combined.groupby('Primary_Pos')[col].transform('median')
        df_combined[col] = df_combined[col].fillna(pos_median).fillna(0)
        
    # 5. Pre-compute Positional Percentiles (0-100) relative strictly to primary position cohort & season
    for col in per90_features:
        pct_col = col.replace('_per90', '_pct')
        df_combined[pct_col] = df_combined.groupby(['Season', 'Primary_Pos'])[col].transform(lambda x: x.rank(pct=True) * 100)
        
    for c in df_combined.select_dtypes(include=['object', 'string']).columns:
        df_combined[c] = df_combined[c].astype(str)
        
    out_path = os.path.join(DATA_DIR, output_filename)
    df_combined.to_parquet(out_path, index=False)
    print(f"Processed {output_filename}: {df_combined.shape}")
    return df_combined


def build_dataset_a():
    print("Building Dataset A (Baseline 3-Season)...")
    df_24 = load_standard_season_file('players_data-2024_2025.csv', '2024-2025')
    df_25 = load_standard_season_file('players_data-2025_2026.csv', '2025-2026')
    df_26 = load_standard_season_file('players_data-2026_2027.csv', '2026-2027')
    df_combined = process_pipeline([df_24, df_25, df_26], 'processed_players_dataset_a.parquet')
    # Keep legacy file updated
    df_combined.to_parquet(os.path.join(DATA_DIR, 'processed_players.parquet'), index=False)
    return df_combined


def build_dataset_b():
    print("Building Dataset B (Enriched 3-Season with Chuong Trinh 2025/2026)...")
    df_24 = load_standard_season_file('players_data-2024_2025.csv', '2024-2025')
    df_25_b = load_and_standardize_2025_b()
    df_26 = load_standard_season_file('players_data-2026_2027.csv', '2026-2027')
    return process_pipeline([df_24, df_25_b, df_26], 'processed_players_dataset_b.parquet')


def fetch_data():
    build_dataset_a()
    build_dataset_b()

if __name__ == '__main__':
    fetch_data()
