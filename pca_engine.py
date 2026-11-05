import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class PCAEngine:
    def __init__(self, n_components=12):
        self.n_components = n_components
        self.features = []
        self.df_pca_global = None
        
        # Global models (General Multi-Season)
        self.global_scalers = {}
        self.global_minmax_scalers = {}
        self.global_pcas = {}
        self.global_loadings = {}
        self.global_dynamic_labels = {}
        
        # Season-specific models (Undiluted)
        self.season_scalers = {}
        self.season_minmax_scalers = {}
        self.season_pcas = {}
        self.season_loadings = {}
        self.season_dynamic_labels = {}
        self.season_features = {}

    def _fit_cohort_pca(self, cohort_df, active_features):
        """Helper to fit StandardScaler, PCA, and MinMaxScaler on a subset with sign alignment."""
        actual_n = min(self.n_components, len(active_features), len(cohort_df))
        if actual_n == 0 or cohort_df.empty:
            zeros = np.zeros((len(cohort_df), self.n_components))
            pca_cols = [f'PC{i+1}' for i in range(self.n_components)]
            return None, None, None, {}, pd.DataFrame(), zeros, zeros
            
        X = cohort_df[active_features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=actual_n, random_state=42)
        pca_feats = pca.fit_transform(X_scaled)
        
        # Fix mathematical inversion (force positive alignment with max variance feature)
        for i in range(actual_n):
            max_idx = np.argmax(np.abs(pca.components_[i]))
            if pca.components_[i, max_idx] < 0:
                pca.components_[i] *= -1
                pca_feats[:, i] *= -1
                
        # Pad up to n_components if fewer components available
        while pca_feats.shape[1] < self.n_components:
            pca_feats = np.column_stack((pca_feats, np.zeros(pca_feats.shape[0])))
            
        mm_scaler = MinMaxScaler()
        pca_feats_mm = mm_scaler.fit_transform(pca_feats)
        
        # Dynamic labels & loadings
        pca_cols = [f'PC{i+1}' for i in range(self.n_components)]
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(actual_n)],
            index=[f.replace('_per90', '') for f in active_features]
        )
        
        labels = {}
        for pc in pca_cols:
            if pc in loadings.columns:
                top_3 = loadings[pc].abs().sort_values(ascending=False).head(3).index.tolist()
                labels[pc] = f"{pc} ({', '.join(top_3)})"
            else:
                labels[pc] = pc
                
        return scaler, pca, mm_scaler, labels, loadings, pca_feats, pca_feats_mm

    def fit_transform_global(self, df):
        """
        Fits both:
        1. Season-Specific Undiluted PCA spaces (ensuring missing metrics in later seasons do not dilute 2024/25).
        2. General Multi-Season Global PCA space across all combined seasons.
        """
        self.features = [c for c in df.columns if c.endswith('_per90')]
        df = df.copy()
        
        pca_cols = [f'PC{i+1}' for i in range(self.n_components)]
        pca_mm_cols = [f'{c}_mm' for c in pca_cols]
        global_pca_cols = [f'PC{i+1}_global' for i in range(self.n_components)]
        global_pca_mm_cols = [f'{c}_mm' for c in global_pca_cols]
        
        # Initialize PC columns
        for c in pca_cols + pca_mm_cols + global_pca_cols + global_pca_mm_cols:
            df[c] = 0.0

        # --- A. Fit General Multi-Season Global PCA ---
        for cohort in ['GK', 'Outfield']:
            cohort_mask = (df['Primary_Pos'] == 'GK') if cohort == 'GK' else (df['Primary_Pos'] != 'GK')
            cohort_df = df[cohort_mask]
            if cohort_df.empty:
                continue
                
            active_feats = [f for f in self.features if cohort_df[f].std() > 1e-6]
            if not active_feats:
                active_feats = self.features
                
            scaler, pca, mm_scaler, labels, loadings, feats, feats_mm = self._fit_cohort_pca(cohort_df, active_feats)
            self.global_scalers[cohort] = scaler
            self.global_pcas[cohort] = pca
            self.global_minmax_scalers[cohort] = mm_scaler
            self.global_dynamic_labels[cohort] = labels
            self.global_loadings[cohort] = loadings
            
            df.loc[cohort_mask, global_pca_cols] = feats
            df.loc[cohort_mask, global_pca_mm_cols] = feats_mm

        # --- B. Fit Season-Specific Undiluted PCA ---
        seasons = df['Season'].unique()
        for season in seasons:
            self.season_scalers[season] = {}
            self.season_pcas[season] = {}
            self.season_minmax_scalers[season] = {}
            self.season_dynamic_labels[season] = {}
            self.season_loadings[season] = {}
            self.season_features[season] = {}
            
            season_mask = (df['Season'] == season)
            
            for cohort in ['GK', 'Outfield']:
                cohort_mask = season_mask & ((df['Primary_Pos'] == 'GK') if cohort == 'GK' else (df['Primary_Pos'] != 'GK'))
                cohort_df = df[cohort_mask]
                if cohort_df.empty:
                    continue
                    
                # Use strictly features with active variance within this season snapshot
                active_feats = [f for f in self.features if cohort_df[f].std() > 1e-6]
                if not active_feats:
                    active_feats = self.features
                self.season_features[season][cohort] = active_feats
                
                scaler, pca, mm_scaler, labels, loadings, feats, feats_mm = self._fit_cohort_pca(cohort_df, active_feats)
                self.season_scalers[season][cohort] = scaler
                self.season_pcas[season][cohort] = pca
                self.season_minmax_scalers[season][cohort] = mm_scaler
                self.season_dynamic_labels[season][cohort] = labels
                self.season_loadings[season][cohort] = loadings
                
                df.loc[cohort_mask, pca_cols] = feats
                df.loc[cohort_mask, pca_mm_cols] = feats_mm
                
        self.df_pca_global = df
        return self.df_pca_global

    def get_similar_players(self, filtered_df, target_idx, top_n=10, mode='season', n_components=None):
        """
        Compute cosine similarity in the selected latent space.
        mode='season': Uses Undiluted Season-Specific PCA embeddings (PC1..PCk)
        mode='global': Uses General Multi-Season PCA embeddings (PC1_global..PCk_global)
        n_components: Number of components to use (defaults to self.n_components)
        """
        target_row = self.df_pca_global.loc[target_idx]
        is_gk = (target_row['Primary_Pos'] == 'GK')
        
        pool = filtered_df[filtered_df['Primary_Pos'] == 'GK'] if is_gk else filtered_df[filtered_df['Primary_Pos'] != 'GK']
        
        k = n_components if n_components is not None else self.n_components
        col_prefix = 'PC'
        col_suffix = '_global' if mode == 'global' else ''
        pca_cols = [f'{col_prefix}{i+1}{col_suffix}' for i in range(k)]
        
        pool_vecs = pool[pca_cols].values
        target_vec = target_row[pca_cols].values.reshape(1, -1)
        
        cos_sim = cosine_similarity(target_vec, pool_vecs)[0]
        sorted_indices = np.argsort(cos_sim)[::-1]
        
        results = []
        rank = 1
        for idx in sorted_indices:
            pool_idx = pool.index[idx]
            if pool_idx == target_idx:
                continue
            match_pct = max(0.0, min(100.0, cos_sim[idx] * 100.0))
            match_row = pool.loc[pool_idx]
            results.append({
                'Rank': rank,
                'Player': match_row['Player'],
                'Squad': match_row['Squad'],
                'Comp': match_row.get('Comp', 'N/A'),
                'Age': match_row['Age'],
                'Match %': f"{match_pct:.2f}%",
                'Index': pool_idx
            })
            rank += 1
            if len(results) >= top_n:
                break
                
        return pd.DataFrame(results)

    def get_loadings(self, cohort='Outfield', season=None, mode='season', n_components=None):
        if mode == 'season' and season in self.season_loadings:
            df_load = self.season_loadings[season].get(cohort, pd.DataFrame())
        else:
            df_load = self.global_loadings.get(cohort, pd.DataFrame())
        if n_components is not None and not df_load.empty:
            cols = [c for c in [f'PC{i+1}' for i in range(n_components)] if c in df_load.columns]
            return df_load[cols]
        return df_load

    def get_dynamic_labels(self, cohort='Outfield', season=None, mode='season', n_components=None):
        if mode == 'season' and season in self.season_dynamic_labels:
            labels = self.season_dynamic_labels[season].get(cohort, {})
        else:
            labels = self.global_dynamic_labels.get(cohort, {})
        if n_components is not None:
            return {k: labels[k] for k in [f'PC{i+1}' for i in range(n_components)] if k in labels}
        return labels
