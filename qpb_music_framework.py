#!/usr/bin/env python3
"""
Quantitative Pattern-Based Framework for Universal Musicological Analysis
==========================================================================

A comprehensive computational musicology framework integrating:
- Synthetic corpus generation and validation (Module 1-2)
- Audio feature extraction (MIR descriptors) (Module 3)
- Comparative statistical analysis (Module 4)
- Advanced clustering with DBSCAN/HDBSCAN (Module 5)
- Supervised classification (Module 6)
- Dimensional visualization (Module 7)
- Harmonic & contrapuntal analysis with historical attribution (Module 8)
- Universal music influence recognition system (Module 9)

ETHICAL FRAMEWORK:
=================
This framework recognizes that all music emerges from complex historical
exchanges, often involving colonialism, migration, resistance, and creative
synthesis. It aims to RESTORE historical complexity rather than impose
Western analytical categories on non-Western traditions.

CORE PRINCIPLES:
- Historical Attribution: Credit innovations to their cultural origins
- Epistemic Humility: Computational hypotheses, not definitive truths
- Collaborative Imperative: Work with cultural experts for non-Western music
- Decolonial Methodology: Recognize power dynamics in musical exchange
- Open Science: Transparent methods, reproducible results, community input

Author: Benseddik Ahmed
Repository: https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis
Version: 4.0 (Integrated)
License: MIT with Mandatory Attribution
DOI: https://doi.org/10.5281/zenodo.17515815

Key Dependencies:
    numpy>=1.21.0, scipy>=1.7.0, pandas>=1.3.0, librosa>=0.9.0,
    scikit-learn>=1.0.0, matplotlib>=3.4.0, music21>=8.0.0,
    networkx>=2.6.0, hdbscan>=0.8.27 (optional)

Citation:
    Benseddik, A. (2025). Quantitative Pattern-Based Framework for 
    Universal Musicological Analysis: An Ethical Approach to Computational 
    Music Analysis with Historical Attribution. GitHub repository.
    https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis

Academic References:
    Floyd, S. A. (1995). The Power of Black Music. Oxford University Press.
    Bohlman, P. V. (2002). World Music: A Very Short Introduction. Oxford.
    Born, G., & Hesmondhalgh, D. (2000). Western Music and Its Others. UC Press.
    Nettl, B. (2015). The Study of Ethnomusicology (3rd ed.). UIUC Press.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, silhouette_score, 
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import networkx as nx

# Music21 imports (for symbolic analysis)
try:
    from music21 import *
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("⚠ music21 not installed. Symbolic analysis (Module 8) will be unavailable.")
    print("  Install via: pip install music21")

# HDBSCAN imports (optional)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠ hdbscan not installed. HDBSCAN clustering will be unavailable.")
    print("  Install via: pip install hdbscan")

warnings.filterwarnings('ignore')

__author__ = "Benseddik Ahmed"
__version__ = "4.0"
__repository__ = "https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis"


# ============================================================================
# FRAMEWORK CONFIGURATION
# ============================================================================

class FrameworkConfig:
    """
    Configuration centrale pour le framework complet.
    """
    
    # Chemins par défaut
    OUTPUT_DIR = Path("./framework_outputs")
    CORPUS_DIR = Path("./corpus")
    RESULTS_DIR = OUTPUT_DIR / "results"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    
    # Paramètres d'analyse
    AUDIO_DURATION = 30  # secondes
    N_MFCC = 13
    
    # Paramètres de clustering
    CLUSTERING_METHODS = ['kmeans', 'dbscan', 'hdbscan']
    DEFAULT_N_CLUSTERS = 3
    
    # Attribution historique
    ENABLE_HISTORICAL_ATTRIBUTION = True
    
    # Verbosité
    VERBOSE = True
    
    @classmethod
    def initialize_directories(cls):
        """Crée les répertoires nécessaires."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.VISUALIZATIONS_DIR.mkdir(exist_ok=True)
        cls.CORPUS_DIR.mkdir(exist_ok=True)


# ============================================================================
# MODULE 1: Synthetic Musical Corpus Generation
# ============================================================================

class BaroqueCorpusGenerator:
    """
    Generator for synthetic musical corpora based on pitch-class theory.
    
    Creates controlled musical sequences with intentionally embedded motifs,
    enabling validation of pattern detection algorithms.
    
    References:
        Forte, A. (1973). The Structure of Atonal Music. Yale University Press.
    """
    
    def __init__(self, seed: int = 42, global_transposition: int = 0) -> None:
        np.random.seed(seed)
        self.c_major_scale = [0, 2, 4, 5, 7, 9, 11]
        self.bach_motif = [10, 9, 0, 11]
        self.c_major_arpeggio = [0, 4, 7, 0]
        self.corpus = []
        self.global_transposition = global_transposition % 12

    def generate_background_passage(self, length: int) -> List[int]:
        passage = []
        for _ in range(length):
            technique = np.random.choice(['scale', 'arpeggio', 'random', 'repeat'])
            
            if technique == 'scale':
                direction = np.random.choice([1, -1])
                start_idx = np.random.randint(0, len(self.c_major_scale))
                for i in range(4):
                    idx = (start_idx + i * direction) % len(self.c_major_scale)
                    passage.append(self.c_major_scale[idx])
                    
            elif technique == 'arpeggio':
                root = np.random.choice(self.c_major_scale)
                passage.extend([root, (root + 4) % 12, (root + 7) % 12, root])
                
            elif technique == 'repeat':
                note = np.random.choice(self.c_major_scale)
                passage.extend([note] * 3)
                
            else:
                passage.append(np.random.choice(self.c_major_scale))
                
        return passage

    def insert_motif_intentionally(self, motif: List[int], frequency: float) -> int:
        motif_len = len(motif)
        expected_count = int((len(self.corpus) / motif_len) * frequency)
        possible_positions = list(range(0, len(self.corpus) - motif_len + 1, motif_len))
        np.random.shuffle(possible_positions)
        positions_to_insert = possible_positions[:expected_count]
        
        for pos in positions_to_insert:
            for i, note in enumerate(motif):
                if pos + i < len(self.corpus):
                    self.corpus[pos + i] = note
                    
        return len(positions_to_insert)

    def transpose_corpus(self) -> None:
        self.corpus = [(note + self.global_transposition) % 12 for note in self.corpus]

    def generate_corpus(self, total_length: int = 1000,
                       bach_frequency: float = 0.03,
                       arpeggio_frequency: float = 0.05) -> Tuple[List[int], Dict]:
        if FrameworkConfig.VERBOSE:
            print("=" * 80)
            print("MODULE 1: SYNTHETIC CORPUS GENERATION")
            print("=" * 80)
        
        self.corpus = self.generate_background_passage(total_length // 4)
        bach_count = self.insert_motif_intentionally(self.bach_motif, bach_frequency)
        arpeggio_count = self.insert_motif_intentionally(self.c_major_arpeggio, arpeggio_frequency)
        
        if self.global_transposition != 0:
            self.transpose_corpus()
        
        note_dist = dict(zip(*np.unique(self.corpus, return_counts=True)))
        note_dist = {int(k): int(v) for k, v in note_dist.items()}

        metadata = {
            'total_length': len(self.corpus),
            'unique_notes': len(set(self.corpus)),
            'note_distribution': note_dist,
            'bach_motif': {
                'sequence': self.bach_motif,
                'inserted_count': bach_count,
                'target_frequency': bach_frequency,
            },
            'c_major_arpeggio': {
                'sequence': self.c_major_arpeggio,
                'inserted_count': arpeggio_count,
                'target_frequency': arpeggio_frequency,
            }
        }

        if FrameworkConfig.VERBOSE:
            print(f"✓ Corpus generated: {len(self.corpus)} pitch-class events")
            print(f"✓ BACH motifs inserted: {bach_count}")
            print(f"✓ Arpeggios inserted: {arpeggio_count}")
        
        return self.corpus, metadata


# ============================================================================
# MODULE 2: Statistical Pattern Analysis
# ============================================================================

class MusicalPatternAnalyzer:
    """
    Analyzer for musical pattern detection with statistical validation.
    
    References:
        Conklin, D., & Witten, I. H. (1995). Multiple viewpoint systems for
        music prediction. Journal of New Music Research, 24(1), 51-73.
    """
    
    def __init__(self, corpus: List[int]) -> None:
        self.corpus = corpus

    def _match_pattern_at(self, corpus: List[int], pattern: List[int], 
                         pos: int, allow_transposition: bool) -> bool:
        plen = len(pattern)
        window = corpus[pos:pos + plen]
        
        if window == pattern:
            return True
            
        if allow_transposition and plen > 1:
            intervals_pat = [pattern[i+1] - pattern[i] for i in range(plen-1)]
            intervals_win = [window[i+1] - window[i] for i in range(plen-1)]
            return intervals_pat == intervals_win
            
        return False

    def find_pattern_occurrences(self, pattern: List[int],
                                allow_transposition: bool = True) -> Dict:
        positions = [i for i in range(len(self.corpus) - len(pattern) + 1)
                    if self._match_pattern_at(self.corpus, pattern, i, allow_transposition)]
        frequency = len(positions) / len(self.corpus) if self.corpus else 0
        
        return {
            'pattern': pattern,
            'occurrences': len(positions),
            'positions': positions,
            'frequency': frequency
        }

    def statistical_test_binomial(self, observed: int, corpus_size: int,
                                  expected_prob: float) -> Dict:
        result = stats.binomtest(observed, corpus_size, expected_prob, alternative='greater')
        obs_prob = observed / corpus_size if corpus_size > 0 else 0
        enrichment = obs_prob / expected_prob if expected_prob > 0 else float('inf')
        
        return {
            'observed': observed,
            'expected': corpus_size * expected_prob,
            'observed_probability': obs_prob,
            'expected_probability': expected_prob,
            'enrichment_ratio': enrichment,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05
        }

    def analyze_pattern(self, pattern: List[int], pattern_name: str) -> Dict:
        if FrameworkConfig.VERBOSE:
            print(f"\n{'=' * 70}")
            print(f"Pattern Analysis: {pattern_name}")
            print('=' * 70)
        
        occ = self.find_pattern_occurrences(pattern)
        
        if FrameworkConfig.VERBOSE:
            print(f"Occurrences: {occ['occurrences']}")
            print(f"Observed frequency: {occ['frequency']:.6f}")

        expected_prob = 1.0 / (len(set(self.corpus)) ** len(pattern))
        binom_res = self.statistical_test_binomial(
            occ['occurrences'], 
            len(self.corpus) - len(pattern) + 1, 
            expected_prob
        )
        
        if FrameworkConfig.VERBOSE:
            print(f"Enrichment ratio: {binom_res['enrichment_ratio']:.2f}×")
            print(f"P-value: {binom_res['p_value']:.6f}")
            print(f"Significant (α=0.05): {binom_res['significant']}")

            interpretation = "✓ Intentional motif detected" if binom_res['significant'] else "✗ No significant pattern"
            print(f"Interpretation: {interpretation}")
        
        return {
            'pattern_name': pattern_name,
            'occurrences': occ,
            'binomial_test': binom_res
        }


# ============================================================================
# MODULE 3: Audio Feature Extraction
# ============================================================================

def extract_audio_features(file_path: str, n_mfcc: int = 13, 
                          duration: Optional[float] = 30) -> Dict[str, float]:
    """
    Extract comprehensive MIR features from audio.
    
    References:
        Peeters, G. (2004). A large set of audio features for sound description.
        McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python.
    """
    y, sr = librosa.load(file_path, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    features = {}
    
    for i in range(n_mfcc):
        features[f'mfcc_mean_{i+1}'] = float(np.mean(mfccs[i]))
        features[f'mfcc_var_{i+1}'] = float(np.var(mfccs[i]))
    
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_var'] = float(np.var(zcr))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
    features['spectral_centroid_var'] = float(np.var(spectral_centroid))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = float(np.mean(rolloff))
    features['rolloff_var'] = float(np.var(rolloff))
    
    return features


def load_corpus_audio_features(corpus_path: str, label: Optional[str] = None) -> pd.DataFrame:
    """Extract features from all audio files in a corpus directory."""
    feature_list = []
    audio_files = [f for f in os.listdir(corpus_path) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if FrameworkConfig.VERBOSE and audio_files:
        print(f"  Extracting features from {len(audio_files)} audio files...")
    
    for filename in audio_files:
        filepath = os.path.join(corpus_path, filename)
        features = extract_audio_features(filepath, n_mfcc=FrameworkConfig.N_MFCC, 
                                         duration=FrameworkConfig.AUDIO_DURATION)
        features['filename'] = filename
        if label:
            features['label'] = label
        feature_list.append(features)
    
    return pd.DataFrame(feature_list)


# ============================================================================
# MODULE 4: Comparative Statistical Analysis
# ============================================================================

def compare_feature_between_groups(df: pd.DataFrame, feature_name: str) -> None:
    """
    Perform Welch's t-test between two musical groups.
    
    References:
        Welch, B. L. (1947). The generalization of "Student's" problem.
    """
    group_labels = df['label'].unique()
    if len(group_labels) != 2:
        if FrameworkConfig.VERBOSE:
            print("⚠ Function optimized for binary comparison only.")
        return
    
    grp1 = df[df['label'] == group_labels[0]][feature_name]
    grp2 = df[df['label'] == group_labels[1]][feature_name]
    
    t_stat, p_val = stats.ttest_ind(grp1, grp2, equal_var=False)
    
    if FrameworkConfig.VERBOSE:
        print(f"\n--- Comparative Analysis: '{feature_name}' ---")
        print(f"Group 1 ({group_labels[0]}): μ={grp1.mean():.4f}, σ={grp1.std():.4f}")
        print(f"Group 2 ({group_labels[1]}): μ={grp2.mean():.4f}, σ={grp2.std():.4f}")
        print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4g}")
        print(f"Significant difference (α=0.05): {'Yes' if p_val < 0.05 else 'No'}")


# ============================================================================
# MODULE 5: Advanced Clustering (K-means, DBSCAN, HDBSCAN)
# ============================================================================

class MusicologicalClusterAnalyzer:
    """
    Advanced clustering system with automatic hyperparameter optimization.
    
    Supports: K-means, DBSCAN, HDBSCAN
    
    References:
        Ester, M., et al. (1996). A density-based algorithm for discovering clusters.
        Campello, R. J., et al. (2013). Density-based clustering based on hierarchical density estimates.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.X = df.drop(columns=['filename', 'label', 'cluster', 'cluster_probability'], errors='ignore')
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.clustering_results = {}
        
        if FrameworkConfig.VERBOSE:
            print(f"✓ Initialized analyzer: {len(df)} samples, {self.X.shape[1]} features")
    
    def optimize_dbscan_eps(self, k: int = 4, percentile_range: Tuple[int, int] = (90, 99)) -> float:
        """Automatically determine optimal epsilon using k-distance graph."""
        if FrameworkConfig.VERBOSE:
            print("\n→ Optimizing DBSCAN epsilon parameter...")
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, _ = neighbors.kneighbors(self.X_scaled)
        
        k_distances = np.sort(distances[:, k-1])
        
        n_samples = len(k_distances)
        start_idx = int(n_samples * percentile_range[0] / 100)
        end_idx = int(n_samples * percentile_range[1] / 100)
        
        gradients = np.gradient(k_distances[start_idx:end_idx])
        elbow_idx = start_idx + np.argmax(gradients)
        optimal_eps = k_distances[elbow_idx]
        
        if FrameworkConfig.VERBOSE:
            print(f"  ✓ Optimal ε: {optimal_eps:.4f}")
        
        return optimal_eps
    
    def optimize_dbscan_min_samples(self, eps: float, range_multiplier: Tuple[int, int] = (2, 10)) -> int:
        """Determine optimal min_samples by maximizing silhouette score."""
        if FrameworkConfig.VERBOSE:
            print("\n→ Optimizing DBSCAN min_samples parameter...")
        
        dim = self.X_scaled.shape[1]
        min_range = max(2, dim * range_multiplier[0] // 10)
        max_range = min(dim * range_multiplier[1] // 10, len(self.df) // 4)
        
        candidates = range(min_range, max_range + 1)
        scores = []
        
        for min_samp in candidates:
            dbscan = DBSCAN(eps=eps, min_samples=min_samp)
            labels = dbscan.fit_predict(self.X_scaled)
            
            mask = labels != -1
            if len(set(labels[mask])) > 1:
                score = silhouette_score(self.X_scaled[mask], labels[mask])
            else:
                score = -1
            scores.append(score)
        
        valid_scores = [(s, m) for s, m in zip(scores, candidates) if s > 0]
        optimal_min_samples = max(valid_scores, key=lambda x: x[0])[1] if valid_scores else min_range
        
        if FrameworkConfig.VERBOSE:
            print(f"  ✓ Optimal min_samples: {optimal_min_samples}")
        
        return optimal_min_samples
    
    def run_kmeans(self, n_clusters: int = 3) -> Dict:
        """Execute K-means clustering."""
        if FrameworkConfig.VERBOSE:
            print(f"\n→ Running K-means (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(self.X_scaled)
        
        metrics = self._compute_clustering_metrics(labels, 'K-means')
        
        result = {
            'method': 'kmeans',
            'labels': labels,
            'n_clusters': n_clusters,
            'model': kmeans,
            'metrics': metrics
        }
        
        self.clustering_results['kmeans'] = result
        return result
    
    def run_dbscan(self, eps: Optional[float] = None, min_samples: Optional[int] = None,
                   auto_optimize: bool = True) -> Dict:
        """Execute DBSCAN clustering with optional optimization."""
        if FrameworkConfig.VERBOSE:
            print(f"\n→ Running DBSCAN clustering...")
        
        if auto_optimize and eps is None:
            eps = self.optimize_dbscan_eps(k=4)
        elif eps is None:
            eps = 0.5
        
        if auto_optimize and min_samples is None:
            min_samples = self.optimize_dbscan_min_samples(eps)
        elif min_samples is None:
            min_samples = 5
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = self._compute_clustering_metrics(labels, 'DBSCAN')
        
        if FrameworkConfig.VERBOSE:
            print(f"  ✓ Clusters detected: {n_clusters}")
            print(f"  ✓ Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        result = {
            'method': 'dbscan',
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples,
            'model': dbscan,
            'metrics': metrics
        }
        
        self.clustering_results['dbscan'] = result
        return result
    
    def run_hdbscan(self, min_cluster_size: Optional[int] = None, min_samples: Optional[int] = None,
                    auto_optimize: bool = True) -> Dict:
        """Execute HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            if FrameworkConfig.VERBOSE:
                print("⚠ HDBSCAN not available, skipping...")
            return {}
        
        if FrameworkConfig.VERBOSE:
            print(f"\n→ Running HDBSCAN clustering...")
        
        if min_cluster_size is None:
            min_cluster_size = max(5, len(self.df) // 20)
        if min_samples is None:
            min_samples = 3
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(self.X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = self._compute_clustering_metrics(labels, 'HDBSCAN')
        
        if FrameworkConfig.VERBOSE:
            print(f"  ✓ Clusters detected: {n_clusters}")
            print(f"  ✓ Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        result = {
            'method': 'hdbscan',
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None,
            'model': clusterer,
            'metrics': metrics
        }
        
        self.clustering_results['hdbscan'] = result
        return result
    
    def _compute_clustering_metrics(self, labels: np.ndarray, method_name: str) -> Dict[str, float]:
        """Compute comprehensive clustering quality metrics."""
        mask = labels != -1
        valid_labels = labels[mask]
        valid_data = self.X_scaled[mask]
        
        metrics = {}
        
        if len(set(valid_labels)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(valid_data, valid_labels)
                metrics['davies_bouldin'] = davies_bouldin_score(valid_data, valid_labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_labels)
            except:
                metrics['silhouette'] = -1
                metrics['davies_bouldin'] = -1
                metrics['calinski_harabasz'] = -1
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        return metrics
    
    def compare_all_methods(self, auto_optimize: bool = True) -> pd.DataFrame:
        """Execute all clustering methods and generate comparative report."""
        if FrameworkConfig.VERBOSE:
            print("\n" + "=" * 80)
            print("MODULE 5: COMPARATIVE CLUSTERING ANALYSIS")
            print("=" * 80)
        
        self.run_kmeans(n_clusters=FrameworkConfig.DEFAULT_N_CLUSTERS)
        self.run_dbscan(auto_optimize=auto_optimize)
        if HDBSCAN_AVAILABLE:
            self.run_hdbscan(auto_optimize=auto_optimize)
        
        comparison = []
        for method, result in self.clustering_results.items():
            if result:
                row = {
                    'Method': method.upper(),
                    'N_Clusters': result['n_clusters'],
                    'Silhouette': result['metrics']['silhouette'],
                    'Davies-Bouldin': result['metrics']['davies_bouldin'],
                    'Calinski-Harabasz': result['metrics']['calinski_harabasz']
                }
                
                if 'n_noise' in result:
                    row['Noise_Points'] = result['n_noise']
                    row['Noise_%'] = f"{result['n_noise']/len(self.df)*100:.1f}%"
                else:
                    row['Noise_Points'] = 0
                    row['Noise_%'] = "0.0%"
                
                comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        
        if FrameworkConfig.VERBOSE:
            print("\n" + "─" * 80)
            print("QUANTITATIVE COMPARISON")
            print("─" * 80)
            print(df_comparison.to_string(index=False))
            print("\nMetric Interpretation:")
            print("  • Silhouette ∈ [-1, 1]: Higher is better (>0.5 excellent)")
            print("  • Davies-Bouldin: Lower is better (<1.0 good)")
            print("  • Calinski-Harabasz: Higher is better (>100 good)")
        
        return df_comparison


# ============================================================================
# MODULE 6: Supervised Classification
# ============================================================================

def perform_audio_classification(df: pd.DataFrame) -> None:
    """
    Train and evaluate Random Forest classifier.
    
    References:
        Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    """
    X = df.drop(columns=['filename', 'label', 'cluster'], errors='ignore')
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                 max_depth=10, min_samples_split=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if FrameworkConfig.VERBOSE:
        print("\n" + "=" * 70)
        print("MODULE 6: CLASSIFICATION REPORT (Random Forest)")
        print("=" * 70)
        print(classification_report(y_test, y_pred, digits=3))


# ============================================================================
# MODULE 7: Dimensional Visualization (PCA)
# ============================================================================

def visualize_pca_projection(df: pd.DataFrame, X_scaled: np.ndarray,
                            color_column: str = 'label',
                            save_path: Optional[Path] = None) -> None:
    """
    Generate 2D PCA projection.
    
    References:
        Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 7))
    groups = df[color_column].unique()
    
    for group in groups:
        subset = df[df[color_column] == group]
        plt.scatter(subset['pca1'], subset['pca2'], 
                   label=str(group), alpha=0.7, s=80, edgecolors='k', linewidth=0.5)
    
    var_explained = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
    plt.title(f'PCA Projection (colored by {color_column})', fontsize=14, fontweight='bold')
    plt.legend(title=color_column.capitalize(), fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if FrameworkConfig.VERBOSE:
            print(f"  ✓ Visualization saved: {save_path}")
    
    plt.show()


# ============================================================================
# ============================================================================
# MODULE 8: Harmonic & Contrapuntal Analysis with Historical Attribution
# ============================================================================
#
# Philosophy: Analyze Western tonal music while explicitly acknowledging
# its debts to African and African-diasporic musical innovations (blue
# notes, syncopation, call-and-response, swing, modal improvisation).
#
# References:
#   Floyd, S. A. (1995). The Power of Black Music. Oxford University Press.
#   Southern, E. (1997). The Music of Black Americans. Norton.
#   Kubik, G. (1999). Africa and the Blues. University Press of Mississippi.
#   Monson, I. (2007). Freedom Sounds. Oxford University Press.
# ============================================================================

# Historical attribution reference data used by Module 8 (and cross-checked
# by Module 9's UniversalInfluenceRegistry).
AFRICAN_DIASPORIC_INNOVATIONS: Dict[str, Dict[str, Any]] = {
    'blue_notes': {
        'description': "Flattened 3rd, 5th, and 7th degrees against a major "
                       "tonal center, a hallmark of African-American blues.",
        'pioneers': ["W.C. Handy", "Bessie Smith", "Robert Johnson"],
        'sources': ["Floyd (1995)", "Southern (1997)"]
    },
    'syncopation': {
        'description': "Rhythmic accentuation of weak beats/offbeats, "
                       "inherited from West African polyrhythmic practice.",
        'pioneers': ["Scott Joplin", "Jelly Roll Morton", "James Brown"],
        'sources': ["Kubik (1999)", "Floyd (1995)"]
    },
    'call_and_response': {
        'description': "Antiphonal structure fundamental to West and "
                       "Central African musical traditions.",
        'pioneers': ["Gospel quartets", "Louis Armstrong"],
        'sources': ["Southern (1997)"]
    },
    'modal_improvisation': {
        'description': "Extended improvisation over static/modal harmony, "
                       "central to post-bop jazz.",
        'pioneers': ["Charlie Parker", "John Coltrane", "Miles Davis"],
        'sources': ["Monson (2007)"]
    },
    'swing_feel': {
        'description': "'Behind the beat' timing and triplet subdivision "
                       "characteristic of the African-diasporic swing era.",
        'pioneers': ["Count Basie", "Duke Ellington"],
        'sources': ["Monson (2007)"]
    },
    'extended_harmony': {
        'description': "Frequent use of 7th/9th chords beyond functional "
                       "resolution, integral to jazz harmonic vocabulary.",
        'pioneers': ["Duke Ellington", "Thelonious Monk"],
        'sources': ["Floyd (1995)"]
    }
}


class HarmonicContrapuntalAnalyzer:
    """
    Module 8: Harmonic & Contrapuntal Analysis with Historical Attribution.

    Analyzes Western tonal scores (MusicXML/MIDI via music21) for:
      - Functional harmony (Roman numerals, tonic/subdominant/dominant)
      - Cadence detection (PAC, IAC, HC, PC, DC)
      - Voice-leading motion between parts (parallel/contrary/oblique/similar)
      - Parallel fifths/octaves (flagged, NOT penalized -- see note below)
      - Blue notes (flat-3, flat-5, flat-7 against a major tonic)
      - Syncopation ratio (via music21 beatStrength)
      - Texture classification (monophonic/homophonic/polyphonic)

    IMPORTANT: "Parallel fifths/octaves" are flagged as classical
    counterpoint would name them, but this is a DESCRIPTIVE label, not
    a value judgment. Jazz, blues, and many non-Western traditions use
    parallel motion intentionally and expressively.
    """

    def __init__(self, score_path: str = None, acknowledge_influences: bool = True,
                assumed_key: Optional[str] = None):
        """
        Args:
            score_path: Path to a music21-readable score file.
            acknowledge_influences: Enable historical attribution.
            assumed_key: Optional manual key override (e.g. "C major",
                "F# minor"). Automatic key-finding (Krumhansl-Schmuckler)
                is unreliable for blues-scale content, where flattened
                3rd/5th/7th degrees are diatonically ambiguous with an
                unrelated major key (e.g. a C blues scale is easily
                misread as Eb major). Use this to force a known tonal
                center when the automatic estimate is implausible.
        """
        if not MUSIC21_AVAILABLE:
            raise ImportError(
                "music21 required for symbolic analysis. Install: pip install music21"
            )

        self.acknowledge_influences = acknowledge_influences

        if score_path:
            self.score = converter.parse(score_path)
            self.score_name = Path(score_path).stem
        else:
            self.score = stream.Score()
            self.score_name = "empty"

        if assumed_key:
            try:
                self.key = key.Key(assumed_key)
            except Exception:
                self.key = key.Key('C')
        else:
            try:
                self.key = self.score.analyze('key')
            except Exception:
                self.key = key.Key('C')

        if FrameworkConfig.VERBOSE:
            print(f"✓ Score loaded: '{self.score_name}'")
            print(f"  Key: {self.key.name} {self.key.mode}")

    # ------------------------------------------------------------------
    # Harmony
    # ------------------------------------------------------------------

    def analyze_harmony(self) -> pd.DataFrame:
        """
        Chordify the score and analyze each simultaneity: Roman numeral,
        harmonic function, inversion, and 7th/9th extensions.
        """
        rows = []
        try:
            chordified = self.score.chordify()
        except Exception:
            return pd.DataFrame()

        for c in chordified.flatten().getElementsByClass('Chord'):
            if len(c.pitches) == 0:
                continue

            entry = {
                'offset': float(c.offset),
                'measure': c.measureNumber if c.measureNumber is not None else -1,
                'pitches': ", ".join(p.nameWithOctave for p in c.pitches),
                'chord_quality': self._chord_quality(c),
                'roman_numeral': None,
                'function': 'Other',
                'has_seventh': c.seventh is not None,
                'has_ninth': self._has_ninth(c),
                'inversion': c.inversion() if c.pitches else 0,
            }

            try:
                rn = roman.romanNumeralFromChord(c, self.key)
                entry['roman_numeral'] = rn.figure
                entry['function'] = self._harmonic_function(rn.scaleDegree)
            except Exception:
                pass

            rows.append(entry)

        return pd.DataFrame(rows)

    @staticmethod
    def _has_ninth(c) -> bool:
        """Check for a 9th extension (music21 Chord has no .ninth attribute)."""
        try:
            return c.getChordStep(9) is not None
        except Exception:
            return False

    @staticmethod
    def _chord_quality(c) -> str:
        try:
            if c.isMajorTriad():
                return 'major'
            if c.isMinorTriad():
                return 'minor'
            if c.isDiminishedTriad():
                return 'diminished'
            if c.isAugmentedTriad():
                return 'augmented'
            if c.isDominantSeventh():
                return 'dominant_seventh'
        except Exception:
            pass
        return 'other'

    @staticmethod
    def _harmonic_function(scale_degree: int) -> str:
        """Classify a Roman-numeral scale degree into T/S/D function."""
        if scale_degree in (1, 6):
            return 'Tonic'
        if scale_degree in (2, 4):
            return 'Subdominant'
        if scale_degree in (5, 7):
            return 'Dominant'
        return 'Other'

    # ------------------------------------------------------------------
    # Blue notes
    # ------------------------------------------------------------------

    def detect_augmented_second_ratio(self) -> float:
        """
        Proportion of consecutive melodic steps (within a single voice)
        that form a genuine augmented 2nd -- distinctive of Hijaz-family
        maqamat and related modes.

        IMPORTANT: an augmented 2nd and a minor 3rd are both 3 semitones
        apart, so they are indistinguishable by raw MIDI/semitone
        difference alone. This uses music21's pitch-spelling-aware
        interval.Interval (generic + chromatic) to tell them apart --
        an earlier semitone-only version falsely fired on blues melodies
        (minor-third blue-note motion), which was caught by testing.
        """
        notes = [n for n in self.score.parts[0].flatten().notes] if \
            hasattr(self.score, 'parts') and len(self.score.parts) > 0 else \
            list(self.score.flatten().notes)

        if len(notes) < 2:
            return 0.0

        aug_second_count = 0
        step_count = 0

        for i in range(1, len(notes)):
            prev_pitches = notes[i - 1].pitches if notes[i - 1].isChord else [notes[i - 1].pitch]
            curr_pitches = notes[i].pitches if notes[i].isChord else [notes[i].pitch]

            if prev_pitches[0].midi == curr_pitches[0].midi:
                continue

            step_count += 1
            try:
                ivl = interval.Interval(prev_pitches[0], curr_pitches[0])
                # Augmented 2nd, ascending or descending
                if ivl.simpleName in ('A2', 'AA1'):
                    aug_second_count += 1
            except Exception:
                continue

        return aug_second_count / step_count if step_count else 0.0

    def detect_blue_notes_context(self) -> Dict[str, Any]:
        """
        Detect flat-3, flat-5, and flat-7 scale degrees against the
        estimated (major) tonic. Only meaningful for major-mode pieces;
        returns an empty result otherwise.
        """
        result = {
            'total_occurrences': 0,
            'frequency': 0.0,
            'patterns': {'flat_third (blue 3rd)': 0,
                        'flat_fifth (blue 5th)': 0,
                        'flat_seventh (blue 7th)': 0},
            'contexts': []
        }

        if self.key.mode != 'major':
            return result

        tonic_pc = self.key.tonic.pitchClass
        blue_intervals = {3: 'flat_third (blue 3rd)',
                          6: 'flat_fifth (blue 5th)',
                          10: 'flat_seventh (blue 7th)'}

        all_notes = list(self.score.flatten().notes)
        total_notes = 0

        for n in all_notes:
            pitches = n.pitches if n.isChord else [n.pitch]
            for p in pitches:
                total_notes += 1
                rel = (p.pitchClass - tonic_pc) % 12
                if rel in blue_intervals:
                    label = blue_intervals[rel]
                    result['patterns'][label] += 1
                    result['total_occurrences'] += 1
                    result['contexts'].append({
                        'offset': float(n.offset),
                        'pitch': p.nameWithOctave,
                        'type': label
                    })

        if total_notes > 0:
            result['frequency'] = result['total_occurrences'] / total_notes

        return result

    # ------------------------------------------------------------------
    # Cadences
    # ------------------------------------------------------------------

    def detect_cadences(self) -> List[Dict[str, Any]]:
        """
        Heuristic cadence detection based on consecutive Roman-numeral
        scale-degree motion in the chordified harmonic reduction.
        """
        cadences = []
        df_harmony = self.analyze_harmony()

        if df_harmony.empty or 'roman_numeral' not in df_harmony:
            return cadences

        chordified = self.score.chordify()
        rn_sequence = []
        for c in chordified.flatten().getElementsByClass('Chord'):
            try:
                rn = roman.romanNumeralFromChord(c, self.key)
                rn_sequence.append((c.offset, c.measureNumber, rn))
            except Exception:
                continue

        for i in range(1, len(rn_sequence)):
            prev_offset, prev_measure, prev_rn = rn_sequence[i - 1]
            curr_offset, curr_measure, curr_rn = rn_sequence[i]

            prev_deg = prev_rn.scaleDegree
            curr_deg = curr_rn.scaleDegree

            cad_type, description = None, None

            if prev_deg == 5 and curr_deg == 1:
                if curr_rn.inversion() == 0 and curr_rn.quality in ('major', 'minor'):
                    cad_type, description = 'PAC', "Perfect Authentic Cadence (V→I)"
                else:
                    cad_type, description = 'IAC', "Imperfect Authentic Cadence (V→I, inverted)"
            elif prev_deg == 4 and curr_deg == 1:
                cad_type, description = 'PC', 'Plagal Cadence ("Amen", IV→I)'
            elif prev_deg == 5 and curr_deg == 6:
                cad_type, description = 'DC', "Deceptive Cadence (V→vi)"
            elif curr_deg == 5:
                cad_type, description = 'HC', "Half Cadence (ends on V)"

            if cad_type:
                cadences.append({
                    'measure': curr_measure if curr_measure is not None else -1,
                    'offset': float(curr_offset),
                    'type': cad_type,
                    'description': description
                })

        return cadences

    # ------------------------------------------------------------------
    # Harmonic rhythm
    # ------------------------------------------------------------------

    def analyze_harmonic_rhythm(self) -> Dict[str, Any]:
        """Average time (in quarter notes) between harmonic changes."""
        chordified = self.score.chordify()
        offsets = [c.offset for c in chordified.flatten().getElementsByClass('Chord')]

        if len(offsets) < 2:
            return {'mean_duration_qn': None, 'interpretation': 'Insufficient data'}

        diffs = np.diff(sorted(offsets))
        mean_dur = float(np.mean(diffs)) if len(diffs) else None

        if mean_dur is None:
            interpretation = 'Insufficient data'
        elif mean_dur < 0.5:
            interpretation = 'Very fast harmonic rhythm (bebop-like)'
        elif mean_dur > 2.0:
            interpretation = 'Slow harmonic rhythm (modal/blues-like)'
        else:
            interpretation = 'Moderate harmonic rhythm'

        return {'mean_duration_qn': mean_dur, 'interpretation': interpretation}

    # ------------------------------------------------------------------
    # Voice leading
    # ------------------------------------------------------------------

    def _voice_offset_pitch_map(self) -> Dict[str, Dict[float, int]]:
        """Build {part_name: {offset: highest_midi_pitch}} for each part."""
        voice_maps = {}
        parts = list(self.score.parts) if hasattr(self.score, 'parts') else []

        if not parts:
            return voice_maps

        for idx, part in enumerate(parts):
            name = part.partName or f"Voice {idx + 1}"
            offset_map = {}
            for n in part.flatten().notes:
                pitches = n.pitches if n.isChord else [n.pitch]
                offset_map[float(n.offset)] = max(p.midi for p in pitches)
            voice_maps[name] = offset_map

        return voice_maps

    def analyze_voice_leading(self) -> pd.DataFrame:
        """
        For each pair of voices/parts, classify melodic motion between
        shared offsets as parallel, contrary, oblique, or similar.
        """
        voice_maps = self._voice_offset_pitch_map()
        voice_names = list(voice_maps.keys())
        rows = []

        for i in range(len(voice_names)):
            for j in range(i + 1, len(voice_names)):
                v1, v2 = voice_names[i], voice_names[j]
                shared_offsets = sorted(set(voice_maps[v1]) & set(voice_maps[v2]))

                for k in range(1, len(shared_offsets)):
                    o_prev, o_curr = shared_offsets[k - 1], shared_offsets[k]
                    d1 = voice_maps[v1][o_curr] - voice_maps[v1][o_prev]
                    d2 = voice_maps[v2][o_curr] - voice_maps[v2][o_prev]

                    motion = self._classify_motion(d1, d2)

                    rows.append({
                        'offset': o_curr,
                        'voice_pair': f"{v1} / {v2}",
                        'motion_type': motion
                    })

        return pd.DataFrame(rows)

    @staticmethod
    def _classify_motion(d1: int, d2: int) -> str:
        if d1 == 0 and d2 == 0:
            return 'static'
        if d1 == 0 or d2 == 0:
            return 'oblique'
        if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
            return 'parallel' if abs(d1) == abs(d2) else 'similar'
        return 'contrary'

    def detect_parallel_fifths_octaves(self) -> List[Dict[str, Any]]:
        """
        Flag consecutive perfect 5ths/octaves between voice pairs.

        NOTE: These "violations" are rules of classical European
        counterpoint. Jazz, blues, and many world music traditions use
        parallel 5ths/octaves intentionally and expressively -- this
        function is descriptive, not evaluative.
        """
        voice_maps = self._voice_offset_pitch_map()
        voice_names = list(voice_maps.keys())
        flags = []

        for i in range(len(voice_names)):
            for j in range(i + 1, len(voice_names)):
                v1, v2 = voice_names[i], voice_names[j]
                shared_offsets = sorted(set(voice_maps[v1]) & set(voice_maps[v2]))

                for k in range(1, len(shared_offsets)):
                    o_prev, o_curr = shared_offsets[k - 1], shared_offsets[k]
                    interval_prev = abs(voice_maps[v1][o_prev] - voice_maps[v2][o_prev]) % 12
                    interval_curr = abs(voice_maps[v1][o_curr] - voice_maps[v2][o_curr]) % 12

                    moved = (voice_maps[v1][o_curr] != voice_maps[v1][o_prev] or
                            voice_maps[v2][o_curr] != voice_maps[v2][o_prev])

                    if moved and interval_prev == interval_curr and interval_curr in (0, 7):
                        label = 'octave/unison' if interval_curr == 0 else 'perfect fifth'
                        flags.append({
                            'offset': o_curr,
                            'voice_pair': f"{v1} / {v2}",
                            'interval': label,
                            'note': ("Descriptive flag only -- common and idiomatic "
                                    "in jazz, blues, rock, and many non-Western "
                                    "traditions.")
                        })

        return flags

    # ------------------------------------------------------------------
    # Texture and syncopation
    # ------------------------------------------------------------------

    def analyze_texture(self) -> str:
        """Rough classification: monophonic / homophonic / polyphonic."""
        parts = list(self.score.parts) if hasattr(self.score, 'parts') else []

        if len(parts) <= 1:
            return 'monophonic'

        rhythm_signatures = []
        for part in parts:
            offsets = tuple(round(float(n.offset), 2) for n in part.flatten().notes)
            rhythm_signatures.append(offsets)

        if len(set(rhythm_signatures)) == 1:
            return 'homophonic'

        overlap_ratios = []
        for i in range(len(rhythm_signatures)):
            for j in range(i + 1, len(rhythm_signatures)):
                a, b = set(rhythm_signatures[i]), set(rhythm_signatures[j])
                if not a or not b:
                    continue
                overlap_ratios.append(len(a & b) / max(len(a | b), 1))

        if overlap_ratios and np.mean(overlap_ratios) > 0.6:
            return 'homophonic'
        return 'polyphonic'

    def compute_syncopation_ratio(self) -> float:
        """
        Fraction of notes attacking on genuine off-beat subdivisions
        (music21 beatStrength < 0.25), a proxy for syncopation.

        NOTE: beatStrength < 0.5 would wrongly include ordinary beat-2/4
        quarter notes in common time (beatStrength == 0.25), which are
        completely normal and NOT syncopated -- that miscalibration was
        caught via a control test (a Bach chorale falsely showing ~58%
        "syncopation"). Only true subdivision-level offbeats (< 0.25)
        are counted here.
        """
        notes = list(self.score.flatten().notes)
        if not notes:
            return 0.0

        weak = 0
        counted = 0
        for n in notes:
            try:
                bs = n.beatStrength
            except Exception:
                continue
            counted += 1
            if bs < 0.25:
                weak += 1

        return weak / counted if counted else 0.0

    # ------------------------------------------------------------------
    # Historical attribution
    # ------------------------------------------------------------------

    def detect_african_diasporic_elements(self, blue_notes: Dict[str, Any],
                                          syncopation_ratio: float,
                                          parallel_motion: List[Dict]) -> Dict[str, Any]:
        """Summarize detected markers with historical attribution text."""
        detected = {}

        if blue_notes.get('total_occurrences', 0) > 0:
            detected['blue_notes'] = AFRICAN_DIASPORIC_INNOVATIONS['blue_notes']

        if syncopation_ratio > 0.35:
            detected['syncopation'] = AFRICAN_DIASPORIC_INNOVATIONS['syncopation']

        if len(parallel_motion) > 0:
            detected['parallel_motion_as_style'] = {
                'description': "Intentional parallel motion, common in jazz/"
                              "blues/gospel voicings rather than a 'flaw'.",
                'pioneers': [],
                'sources': ["Floyd (1995)"]
            }

        return detected

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Run the full Module 8 pipeline and return a results dictionary
        consumed by Module 9's influence detector and by the master
        UniversalMusicologicalFramework.
        """
        if FrameworkConfig.VERBOSE:
            print("\n" + "═" * 80)
            print("MODULE 8: HARMONIC & CONTRAPUNTAL ANALYSIS")
            print("═" * 80)

        df_harmony = self.analyze_harmony()
        cadences = self.detect_cadences()
        blue_notes = self.detect_blue_notes_context()
        df_voice_leading = self.analyze_voice_leading()
        parallel_motion = self.detect_parallel_fifths_octaves()
        harmonic_rhythm = self.analyze_harmonic_rhythm()
        texture = self.analyze_texture()
        syncopation_ratio = self.compute_syncopation_ratio()
        augmented_second_ratio = self.detect_augmented_second_ratio()

        historical_attribution = {}
        if self.acknowledge_influences:
            historical_attribution = self.detect_african_diasporic_elements(
                blue_notes, syncopation_ratio, parallel_motion
            )

        results = {
            'score_name': self.score_name,
            'key': f"{self.key.name} {self.key.mode}",
            'harmony': df_harmony,
            'cadences': cadences,
            'voice_leading': df_voice_leading,
            'parallel_motion': parallel_motion,
            'blue_notes': blue_notes,
            'augmented_second_ratio': augmented_second_ratio,
            'harmonic_rhythm': harmonic_rhythm,
            'texture': texture,
            'syncopation_ratio': syncopation_ratio,
            'historical_attribution': historical_attribution
        }

        if FrameworkConfig.VERBOSE:
            print(f"✓ Harmonic events: {len(df_harmony)}")
            print(f"✓ Cadences detected: {len(cadences)}")
            print(f"✓ Blue notes: {blue_notes['total_occurrences']}")
            print(f"✓ Syncopation ratio: {syncopation_ratio:.2%}")
            print(f"✓ Texture: {texture}")
            if historical_attribution:
                print("\n📜 HISTORICAL ATTRIBUTION:")
                for key_name, info in historical_attribution.items():
                    print(f"  • {key_name}: {info['description']}")
                    if info.get('pioneers'):
                        print(f"    Pioneers: {', '.join(info['pioneers'])}")

        return results


# ============================================================================
# MODULE 9: Universal Influence Attribution
# ============================================================================
#
# Philosophy: Do NOT attempt to "analyze all world music" (impossible and
# colonial). Instead, RECOGNIZE that music emerges from cross-cultural
# exchange, and document that exchange with academic citations, explicit
# certainty levels, and explicit power dynamics (colonial, trade,
# voluntary exchange, resistance, etc.).
#
# All detections from this module are HYPOTHESES requiring validation by
# musicologists, historians, and tradition-bearers -- never definitive
# historical claims.
# ============================================================================

@dataclass
class MusicalInfluence:
    """Represents a documented historical musical influence."""
    source_tradition: str
    target_tradition: str
    period: str
    mechanism: str
    musical_elements: List[str]
    key_figures: List[str] = field(default_factory=list)
    academic_sources: List[str] = field(default_factory=list)
    certainty: str = 'established'          # 'established' | 'probable' | 'speculative'
    bidirectional: bool = False
    power_dynamics: Optional[str] = None    # 'colonial', 'trade', 'voluntary_exchange',
                                             # 'colonial_resistance', 'postcolonial_retention'


class UniversalInfluenceRegistry:
    """
    Registry of documented cross-cultural musical influences.

    Every entry carries academic sources and an explicit certainty level.
    This registry is intentionally curated (quality over quantity) and is
    meant to grow through collaboration with domain experts -- see
    add_influence() and the CONTRIBUTING guidelines.
    """

    def __init__(self):
        self.influences: List[MusicalInfluence] = []
        self._initialize_documented_influences()

    def _initialize_documented_influences(self):
        entries = [
            MusicalInfluence(
                source_tradition="West African Polyrhythm",
                target_tradition="African-American Blues/Jazz",
                period="1619-1900 CE",
                mechanism="Forced migration, cultural retention under slavery",
                musical_elements=["Polyrhythm", "Call-response", "Blue notes", "Syncopation"],
                key_figures=["W.C. Handy", "Louis Armstrong", "Charlie Parker"],
                academic_sources=["Floyd (1995)", "Southern (1997)", "Kubik (1999)"],
                certainty='established',
                power_dynamics='colonial_resistance'
            ),
            MusicalInfluence(
                source_tradition="African-American Jazz",
                target_tradition="European Classical Music (20th c.)",
                period="1910-1950 CE",
                mechanism="Touring musicians, recordings, Parisian jazz scene",
                musical_elements=["Syncopation", "Blue notes", "Swing", "Extended harmonies"],
                key_figures=["Debussy", "Ravel", "Stravinsky", "Milhaud"],
                academic_sources=["Howat (2000)", "Taruskin (2005)"],
                certainty='established',
                bidirectional=False,
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Arabic/Persian Music Theory",
                target_tradition="Medieval European Music",
                period="711-1492 CE (Al-Andalus)",
                mechanism="Cultural contact in Moorish Iberia",
                musical_elements=["Lute/guitar ancestry", "Modal systems",
                                  "Melismatic vocals", "Rhythmic modes"],
                key_figures=["Ziryab", "Al-Farabi", "Ibn Bajja"],
                academic_sources=["Farmer (1988)"],
                certainty='probable',
                power_dynamics='trade'
            ),
            MusicalInfluence(
                source_tradition="Arabic Maqam",
                target_tradition="Spanish Flamenco",
                period="711-1492 CE and after",
                mechanism="Cultural continuity post-Reconquista, Romani mediation",
                musical_elements=["Phrygian mode", "Melismatic ornamentation",
                                  "Microtonal inflections", "Modal improvisation"],
                key_figures=["Antonio Chacón", "Paco de Lucía"],
                academic_sources=["Schuyler (1978)", "Washabaugh (2012)"],
                certainty='probable',
                power_dynamics='postcolonial_retention'
            ),
            MusicalInfluence(
                source_tradition="Ottoman/Turkish Makam",
                target_tradition="Balkan Folk Music",
                period="1299-1922 CE",
                mechanism="Ottoman administrative and cultural presence",
                musical_elements=["Makam system", "Asymmetric meters (7/8, 9/8, 11/8)",
                                  "Ornamentation", "Saz, zurna, davul"],
                key_figures=[],
                academic_sources=["Pennanen (2004)"],
                certainty='established',
                power_dynamics='colonial'
            ),
            MusicalInfluence(
                source_tradition="Hindustani Classical Music",
                target_tradition="1960s Western Rock",
                period="1960-1970 CE",
                mechanism="Direct study, counterculture cross-cultural exchange",
                musical_elements=["Sitar", "Drone", "Raga", "Tala", "Modal improvisation"],
                key_figures=["Ravi Shankar", "George Harrison"],
                academic_sources=["Lavezzoli (2006)", "Farrell (1997)"],
                certainty='established',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Indian Classical Music",
                target_tradition="American Minimalism",
                period="1960-1980 CE",
                mechanism="Direct study with Indian masters",
                musical_elements=["Drone", "Slow-developing process", "Just intonation"],
                key_figures=["Terry Riley", "La Monte Young", "Pandit Pran Nath"],
                academic_sources=["Lavezzoli (2006)"],
                certainty='established',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Javanese Gamelan",
                target_tradition="French Impressionism",
                period="1889-1920 CE",
                mechanism="1889 Paris Exposition Universelle",
                musical_elements=["Slendro/pelog scales", "Parallel motion",
                                  "Stratified texture", "Metallic timbres"],
                key_figures=["Debussy", "Ravel"],
                academic_sources=["Lesure & Howat (1999)"],
                certainty='established',
                power_dynamics='colonial'
            ),
            MusicalInfluence(
                source_tradition="Japanese Traditional Music",
                target_tradition="20th-Century Avant-Garde",
                period="1950-1980 CE",
                mechanism="Post-war cultural exchange, Zen influence",
                musical_elements=["Ma (silence as structure)", "Heterophonic texture",
                                  "Breath-based phrasing", "Non-teleological form"],
                key_figures=["John Cage", "Stockhausen", "Toru Takemitsu"],
                academic_sources=["Kostelanetz (2003)"],
                certainty='probable',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Chinese Pentatonicism",
                target_tradition="Global Popular Music",
                period="Ancient-Present",
                mechanism="Diffuse, long-term cultural diffusion",
                musical_elements=["Pentatonic scales", "Ornamentation", "Sliding pitches"],
                key_figures=["Lou Harrison", "Tan Dun"],
                academic_sources=["Liu & Monroe (1989)"],
                certainty='speculative',
                power_dynamics='trade'
            ),
            MusicalInfluence(
                source_tradition="Cuban Son/Rumba",
                target_tradition="American Jazz (Latin Jazz)",
                period="1940-1970 CE",
                mechanism="Musician migration and collaboration (NYC)",
                musical_elements=["Clave", "Montuno", "Tumbao", "Polyrhythmic percussion"],
                key_figures=["Dizzy Gillespie", "Chano Pozo", "Machito", "Tito Puente"],
                academic_sources=["Roberts (1999)"],
                certainty='established',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Brazilian Samba/Bossa Nova",
                target_tradition="Cool Jazz",
                period="1950-1970 CE",
                mechanism="Recordings, touring musicians (Getz/Gilberto)",
                musical_elements=["Syncopated rhythms", "Harmonic sophistication",
                                  "Nylon-string guitar textures"],
                key_figures=["Antonio Carlos Jobim", "João Gilberto", "Stan Getz"],
                academic_sources=["Castro (2000)"],
                certainty='established',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Jamaican Reggae/Ska",
                target_tradition="British Punk/Post-Punk",
                period="1970-1990 CE",
                mechanism="Caribbean diaspora communities in the UK",
                musical_elements=["Offbeat emphasis", "Heavy bass", "Dub production techniques"],
                key_figures=["Bob Marley", "Lee 'Scratch' Perry", "The Clash", "The Police"],
                academic_sources=["Hebdige (1987)"],
                certainty='established',
                power_dynamics='postcolonial_retention'
            ),
            MusicalInfluence(
                source_tradition="West African Highlife",
                target_tradition="Afrobeat",
                period="1950-1980 CE",
                mechanism="Direct musical evolution within Ghana/Nigeria",
                musical_elements=["Guitar-band idiom with traditional percussion",
                                  "Call-response", "Extended groove sections"],
                key_figures=["Fela Kuti", "E.T. Mensah", "Tony Allen"],
                academic_sources=["Collins (1992)"],
                certainty='established',
                power_dynamics='postcolonial_retention'
            ),
            MusicalInfluence(
                source_tradition="Klezmer",
                target_tradition="American Jazz / Tin Pan Alley",
                period="1880-1940 CE",
                mechanism="Jewish immigration to the United States",
                musical_elements=["Modal scales", "Krekhts ornaments", "Clarinet virtuosity"],
                key_figures=["Benny Goodman", "George Gershwin", "Dave Tarras"],
                academic_sources=["Slobin (2000)"],
                certainty='probable',
                power_dynamics='voluntary_exchange'
            ),
            MusicalInfluence(
                source_tradition="Native American Music",
                target_tradition="American Experimental Music",
                period="1920-Present",
                mechanism="Cultural exchange (with significant ethical caveats)",
                musical_elements=["Vocables", "Pentatonicism", "Drone", "Nature soundscapes"],
                key_figures=["R. Carlos Nakai", "Buffy Sainte-Marie"],
                academic_sources=["Nettl (1989)"],
                certainty='speculative',
                power_dynamics='colonial'
            ),
        ]

        self.influences.extend(entries)

        if FrameworkConfig.VERBOSE:
            print(f"✓ Influence registry initialized: {len(self.influences)} documented influences")

    # ------------------------------------------------------------------

    def add_influence(self, influence: MusicalInfluence) -> None:
        """Add a new (ideally expert-reviewed) influence to the registry."""
        self.influences.append(influence)

    def search_influences(self, keyword: str) -> List[MusicalInfluence]:
        """Case-insensitive search across source/target tradition names."""
        keyword_lower = keyword.lower()
        return [inf for inf in self.influences
                if keyword_lower in f"{inf.source_tradition} {inf.target_tradition}".lower()]

    def get_by_certainty(self, certainty: str) -> List[MusicalInfluence]:
        """Filter influences by certainty level ('established'/'probable'/'speculative')."""
        return [inf for inf in self.influences if inf.certainty == certainty]

    def export_registry(self, output_path: str) -> None:
        """Export the full registry as JSON for external use/review."""
        data = [
            {
                'source_tradition': inf.source_tradition,
                'target_tradition': inf.target_tradition,
                'period': inf.period,
                'mechanism': inf.mechanism,
                'musical_elements': inf.musical_elements,
                'key_figures': inf.key_figures,
                'academic_sources': inf.academic_sources,
                'certainty': inf.certainty,
                'bidirectional': inf.bidirectional,
                'power_dynamics': inf.power_dynamics
            }
            for inf in self.influences
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if FrameworkConfig.VERBOSE:
            print(f"✓ Registry exported: {output_path} ({len(data)} entries)")


def _seventh_chord_ratio(analysis_results: Dict[str, Any]) -> float:
    """Proportion of harmonic events containing a 7th (0.0 if no data)."""
    df = analysis_results.get('harmony', pd.DataFrame())
    if df is None or len(df) == 0 or 'has_seventh' not in df:
        return 0.0
    return float(df['has_seventh'].mean())


def _ninth_chord_ratio(analysis_results: Dict[str, Any]) -> float:
    """Proportion of harmonic events containing a 9th (0.0 if no data)."""
    df = analysis_results.get('harmony', pd.DataFrame())
    if df is None or len(df) == 0 or 'has_ninth' not in df:
        return 0.0
    return float(df['has_ninth'].mean())


def _nonfunctional_harmony_ratio(analysis_results: Dict[str, Any]) -> float:
    """
    Proportion of harmonic events classified as 'Other' function
    (neither clearly Tonic, Subdominant, nor Dominant) -- a rough proxy
    for modal/non-functional harmony (parallel planing chords, quartal
    harmony, etc.) as opposed to classical tonal harmony.
    """
    df = analysis_results.get('harmony', pd.DataFrame())
    if df is None or len(df) == 0 or 'function' not in df:
        return 0.0
    return float((df['function'] == 'Other').mean())


class UniversalInfluenceDetector:
    """
    Detects potential historical influences from Module 8 analysis results.

    CRITICAL: All outputs are computational HYPOTHESES, not definitive
    historical conclusions. They require validation by musicologists,
    historians, and practitioners of the traditions in question.
    """

    # Simple rule set: influence_key -> registry search keyword + the
    # Module-8 signals (as boolean predicates over the results dict) that
    # contribute evidence, each with a confidence weight.
    DETECTION_RULES: Dict[str, Dict[str, Any]] = {
        'african_diasporic_blues_jazz': {
            'registry_keyword': 'African-American',
            'min_markers': 2,
            'checks': [
                (lambda r: r.get('blue_notes', {}).get('total_occurrences', 0) > 0, 0.4, 'blue_notes'),
                (lambda r: r.get('syncopation_ratio', 0) > 0.35, 0.3, 'high_syncopation'),
                (lambda r: _seventh_chord_ratio(r) > 0.30, 0.2, 'dense_seventh_chords'),
                (lambda r: len(r.get('parallel_motion', [])) > 2, 0.1, 'frequent_parallel_motion'),
            ]
        },
        'jazz_influence_on_classical': {
            'registry_keyword': 'Jazz',
            'min_markers': 2,
            'checks': [
                (lambda r: r.get('blue_notes', {}).get('total_occurrences', 0) > 0, 0.35, 'blue_notes'),
                (lambda r: _ninth_chord_ratio(r) > 0.15, 0.35, 'dense_ninth_chords'),
                (lambda r: r.get('harmonic_rhythm', {}).get('mean_duration_qn') is not None
                          and r['harmonic_rhythm']['mean_duration_qn'] > 2.0, 0.3, 'modal_static_harmony'),
            ]
        },
        'arabic_andalusian': {
            'registry_keyword': 'Arabic',
            'min_markers': 2,
            'checks': [
                (lambda r: r.get('augmented_second_ratio', 0) > 0.1, 0.5, 'augmented_second_motion'),
                (lambda r: r.get('blue_notes', {}).get('patterns', {}).get(
                    'flat_third (blue 3rd)', 0) > 0, 0.3, 'lowered_third'),
                (lambda r: r.get('texture') == 'monophonic', 0.15, 'monophonic_texture_supporting'),
            ]
        },
        'latin_caribbean': {
            'registry_keyword': 'Cuban',
            'min_markers': 2,
            'checks': [
                (lambda r: r.get('syncopation_ratio', 0) > 0.5, 0.5, 'very_high_syncopation'),
                (lambda r: len(r.get('parallel_motion', [])) > 3, 0.3, 'frequent_parallel_voicings'),
                (lambda r: _seventh_chord_ratio(r) > 0.30, 0.2, 'dense_seventh_chords'),
            ]
        },
        'gamelan_impressionism': {
            'registry_keyword': 'Gamelan',
            'min_markers': 2,
            'checks': [
                (lambda r: len(r.get('parallel_motion', [])) > 3, 0.4, 'frequent_parallel_motion'),
                (lambda r: r.get('texture') == 'homophonic', 0.3, 'stratified_homophonic_texture'),
                (lambda r: r.get('harmonic_rhythm', {}).get('mean_duration_qn') is not None
                          and r['harmonic_rhythm']['mean_duration_qn'] >= 1.0, 0.2, 'sustained_harmonic_rhythm'),
            ]
        },
    }

    def __init__(self):
        self.registry = UniversalInfluenceRegistry()

    def detect_influences(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score each rule in DETECTION_RULES against the Module 8 results
        and return influences that cross a minimal confidence threshold.
        """
        if FrameworkConfig.VERBOSE:
            print("\n" + "=" * 80)
            print("MODULE 9: UNIVERSAL INFLUENCE DETECTION")
            print("=" * 80)
            print("\n⚠️  IMPORTANT: These are HYPOTHESES requiring expert validation\n")

        detected_influences = {}

        for influence_key, rule in self.DETECTION_RULES.items():
            confidence = 0.0
            markers = []

            for check_fn, weight, marker_name in rule['checks']:
                try:
                    if check_fn(analysis_results):
                        confidence += weight
                        markers.append(marker_name)
                except Exception:
                    continue

            confidence = min(confidence, 1.0)
            min_markers = rule.get('min_markers', 1)

            if confidence >= 0.3 and len(markers) >= min_markers:
                if confidence > 0.7:
                    interpretation = "Strong indicators of influence"
                elif confidence > 0.5:
                    interpretation = "Moderate indicators of influence"
                else:
                    interpretation = "Weak indicators of influence"

                detected_influences[influence_key] = {
                    'confidence': confidence,
                    'detected_markers': markers,
                    'historical_influences': self.registry.search_influences(rule['registry_keyword']),
                    'interpretation': interpretation
                }

        if FrameworkConfig.VERBOSE:
            if detected_influences:
                print(f"✓ {len(detected_influences)} potential influence(s) detected")
                for inf_name, data in detected_influences.items():
                    print(f"  • {inf_name}: {data['interpretation']} ({data['confidence']:.0%})")
            else:
                print("✓ No significant influence markers detected")

        return detected_influences

    def generate_influence_report(self, detected_influences: Dict, score_name: str) -> str:
        """Generate a human-readable influence detection report."""
        report = []
        report.append(f"\n{'=' * 80}")
        report.append(f"INFLUENCE DETECTION REPORT: {score_name}")
        report.append(f"{'=' * 80}")

        if detected_influences:
            for inf_name, data in detected_influences.items():
                report.append(f"\n{inf_name.upper()}: {data['interpretation']} "
                             f"({data['confidence']:.0%} confidence)")
                report.append(f"  Detected markers: {', '.join(data['detected_markers'])}")

                for hist_inf in data['historical_influences'][:2]:
                    report.append(f"\n  → {hist_inf.source_tradition} → {hist_inf.target_tradition}")
                    report.append(f"    Period: {hist_inf.period}")
                    report.append(f"    Mechanism: {hist_inf.mechanism}")
                    if hist_inf.key_figures:
                        report.append(f"    Key figures: {', '.join(hist_inf.key_figures)}")
                    if hist_inf.academic_sources:
                        report.append(f"    Source: {hist_inf.academic_sources[0]}")
        else:
            report.append("\nNo significant influences detected.")

        report.append(f"\n{'-' * 80}")
        report.append("CRITICAL REMINDER: These are computational hypotheses, not")
        report.append("historical facts. Validation by domain experts is required.")
        report.append(f"{'=' * 80}")

        return "\n".join(report)


class InfluenceNetworkVisualizer:
    """
    Visualizes the UniversalInfluenceRegistry as a directed graph using
    networkx: nodes are traditions, edges are documented influences.
    """

    CERTAINTY_COLORS = {
        'established': 'tab:blue',
        'probable': 'tab:orange',
        'speculative': 'tab:red'
    }

    def __init__(self, registry: UniversalInfluenceRegistry):
        self.registry = registry
        self.graph = self._build_graph()

    def _build_graph(self) -> "nx.DiGraph":
        G = nx.DiGraph()
        for inf in self.registry.influences:
            G.add_node(inf.source_tradition)
            G.add_node(inf.target_tradition)
            G.add_edge(
                inf.source_tradition, inf.target_tradition,
                period=inf.period,
                mechanism=inf.mechanism,
                certainty=inf.certainty,
                power_dynamics=inf.power_dynamics,
                elements=", ".join(inf.musical_elements)
            )
        return G

    def visualize_full_network(self, save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (16, 12)) -> None:
        """Draw the complete influence network, colored by certainty."""
        pos = nx.spring_layout(self.graph, seed=42, k=0.9)
        edge_colors = [
            self.CERTAINTY_COLORS.get(self.graph[u][v]['certainty'], 'gray')
            for u, v in self.graph.edges()
        ]

        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightsteelblue',
                              node_size=1800, alpha=0.9)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors,
                              arrows=True, arrowsize=15, width=1.5, alpha=0.7)

        legend_handles = [
            plt.Line2D([0], [0], color=color, lw=2, label=cert.capitalize())
            for cert, color in self.CERTAINTY_COLORS.items()
        ]
        plt.legend(handles=legend_handles, loc='upper left', title='Certainty')
        plt.title("Universal Influence Registry — Cross-Cultural Musical Influence Network",
                 fontsize=13, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if FrameworkConfig.VERBOSE:
                print(f"  ✓ Network visualization saved: {save_path}")

        plt.show()

    def visualize_tradition_influences(self, tradition_name: str,
                                       direction: str = 'both',
                                       save_path: Optional[str] = None) -> None:
        """Draw the subgraph of influences into/out of a single tradition."""
        if tradition_name not in self.graph:
            print(f"⚠ Tradition '{tradition_name}' not found in registry.")
            return

        if direction == 'in':
            edges = list(self.graph.in_edges(tradition_name))
        elif direction == 'out':
            edges = list(self.graph.out_edges(tradition_name))
        else:
            edges = list(self.graph.in_edges(tradition_name)) + \
                    list(self.graph.out_edges(tradition_name))

        subgraph = self.graph.edge_subgraph(edges).copy() if edges else nx.DiGraph()

        if len(subgraph) == 0:
            print(f"⚠ No {direction}-edges found for '{tradition_name}'.")
            return

        pos = nx.spring_layout(subgraph, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightsteelblue', node_size=2000)
        nx.draw_networkx_labels(subgraph, pos, font_size=9)
        nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowsize=18)
        plt.title(f"Influences ({direction}) — {tradition_name}", fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def export_network(self, output_path: str) -> None:
        """Export the network as GEXF for external tools (e.g. Gephi)."""
        nx.write_gexf(self.graph, output_path)
        if FrameworkConfig.VERBOSE:
            print(f"✓ Network exported: {output_path}")
# ============================================================================
# INTEGRATED PIPELINE: Complete Analysis Workflow
# ============================================================================

class UniversalMusicologicalFramework:
    """
    Master class orchestrating all framework modules.
    
    Provides unified interface for:
    1. Synthetic corpus analysis (Modules 1-2)
    2. Audio corpus analysis (Modules 3-7)
    3. Symbolic music analysis (Module 8)
    4. Universal influence detection (Module 9)
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize the complete framework.
        
        Args:
            config: Optional custom configuration
        """
        self.config = config or FrameworkConfig()
        self.config.initialize_directories()
        
        # Module states
        self.synthetic_corpus = None
        self.synthetic_metadata = None
        self.audio_features_df = None
        self.clustering_analyzer = None
        self.symbolic_analyzer = None
        self.influence_detector = UniversalInfluenceDetector()
        
        # Results storage
        self.results = {
            'synthetic_analysis': {},
            'audio_analysis': {},
            'symbolic_analysis': {},
            'influence_analysis': {}
        }
        
        if FrameworkConfig.VERBOSE:
            self._print_welcome_banner()
    
    def _print_welcome_banner(self):
        """Print framework welcome banner."""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 10 + "QUANTITATIVE PATTERN-BASED FRAMEWORK v4.0" + " " * 27 + "█")
        print("█" + " " * 15 + "Universal Musicological Analysis" + " " * 32 + "█")
        print("█" + " " * 20 + "with Historical Attribution" + " " * 33 + "█")
        print("█" + " " * 78 + "█")
        print("█" + " " * 25 + "Author: Benseddik Ahmed" + " " * 31 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80 + "\n")
        
        print("Framework Modules:")
        print("  [1-2] Synthetic Corpus Generation & Pattern Analysis")
        print("  [3-4] Audio Feature Extraction & Statistical Comparison")
        print("  [5-7] Advanced Clustering, Classification & Visualization")
        print("  [8]   Harmonic & Contrapuntal Analysis with Historical Attribution")
        print("  [9]   Universal Music Influence Recognition System")
        print("\n" + "─" * 80 + "\n")
    
    # ========================================================================
    # PART 1: SYNTHETIC CORPUS ANALYSIS
    # ========================================================================
    
    def run_synthetic_corpus_analysis(self, corpus_length: int = 1000,
                                     bach_frequency: float = 0.03,
                                     arpeggio_frequency: float = 0.05) -> Dict:
        """
        Execute synthetic corpus generation and pattern analysis (Modules 1-2).
        
        Args:
            corpus_length: Total length of corpus in pitch-class events
            bach_frequency: Target frequency for BACH motif insertion
            arpeggio_frequency: Target frequency for arpeggio insertion
            
        Returns:
            Dictionary with corpus, metadata, and pattern analysis results
        """
        if FrameworkConfig.VERBOSE:
            print("\n" + "▶" * 40)
            print("PART 1: SYNTHETIC CORPUS GENERATION & VALIDATION")
            print("▶" * 40 + "\n")
        
        # Module 1: Generate corpus
        generator = BaroqueCorpusGenerator(seed=42)
        self.synthetic_corpus, self.synthetic_metadata = generator.generate_corpus(
            total_length=corpus_length,
            bach_frequency=bach_frequency,
            arpeggio_frequency=arpeggio_frequency
        )
        
        # Module 2: Pattern analysis
        analyzer = MusicalPatternAnalyzer(self.synthetic_corpus)
        
        test_patterns = {
            'BACH Motif (B-A-C-H)': [10, 9, 0, 11],
            'C Major Arpeggio': [0, 4, 7, 0],
            'Ascending Scale Fragment': [0, 2, 4, 5],
            'Random Control Pattern': [3, 8, 1, 6]
        }
        
        pattern_results = {}
        for name, pattern in test_patterns.items():
            pattern_results[name] = analyzer.analyze_pattern(pattern, name)
        
        self.results['synthetic_analysis'] = {
            'corpus': self.synthetic_corpus,
            'metadata': self.synthetic_metadata,
            'pattern_results': pattern_results
        }
        
        # Save results
        output_path = self.config.RESULTS_DIR / 'synthetic_corpus_analysis.json'
        with open(output_path, 'w') as f:
            json_safe_results = {
                'metadata': self.synthetic_metadata,
                'pattern_results': {
                    k: {
                        'occurrences': int(v['occurrences']['occurrences']),
                        'frequency': float(v['occurrences']['frequency']),
                        'p_value': float(v['binomial_test']['p_value']),
                        'significant': bool(v['binomial_test']['significant'])
                    }
                    for k, v in pattern_results.items()
                }
            }
            json.dump(json_safe_results, f, indent=2)
        
        if FrameworkConfig.VERBOSE:
            print(f"\n✓ Results saved: {output_path}")
        
        return self.results['synthetic_analysis']
    
    # ========================================================================
    # PART 2: AUDIO CORPUS ANALYSIS
    # ========================================================================
    
    def run_audio_corpus_analysis(self, corpus_directories: Dict[str, str],
                                  run_classification: bool = True,
                                  run_clustering: bool = True,
                                  compare_features: Optional[List[str]] = None) -> Dict:
        """
        Execute complete audio corpus analysis (Modules 3-7).
        
        Args:
            corpus_directories: Dict mapping labels to directory paths
            run_classification: Whether to run supervised classification
            run_clustering: Whether to run clustering analysis
            compare_features: List of features to compare between groups
            
        Returns:
            Dictionary with audio analysis results
        """
        if FrameworkConfig.VERBOSE:
            print("\n" + "▶" * 40)
            print("PART 2: AUDIO CORPUS EXTRACTION & ANALYSIS")
            print("▶" * 40 + "\n")
        
        # Module 3: Feature extraction
        if FrameworkConfig.VERBOSE:
            print("=" * 80)
            print("MODULE 3: AUDIO FEATURE EXTRACTION")
            print("=" * 80)
        
        dataframes = []
        for label, directory in corpus_directories.items():
            if not os.path.isdir(directory):
                if FrameworkConfig.VERBOSE:
                    print(f"⚠ Directory '{directory}' not found. Skipping '{label}' corpus.")
                continue
            
            if FrameworkConfig.VERBOSE:
                print(f"\n→ Processing '{label}' corpus from: {directory}")
            
            df_subset = load_corpus_audio_features(directory, label=label)
            dataframes.append(df_subset)
        
        if not dataframes:
            if FrameworkConfig.VERBOSE:
                print("\n⚠ NO AUDIO CORPUS DETECTED")
                print("To enable audio analysis, create corpus directories and add .wav files.")
            return {}
        
        self.audio_features_df = pd.concat(dataframes, ignore_index=True)
        
        if FrameworkConfig.VERBOSE:
            print(f"\n✓ Total audio excerpts analyzed: {len(self.audio_features_df)}")
        
        # Save extracted features
        features_path = self.config.RESULTS_DIR / 'audio_features.csv'
        self.audio_features_df.to_csv(features_path, index=False)
        if FrameworkConfig.VERBOSE:
            print(f"✓ Features saved: {features_path}")
        
        # Module 4: Statistical comparison
        if compare_features and len(corpus_directories) == 2:
            if FrameworkConfig.VERBOSE:
                print("\n" + "=" * 80)
                print("MODULE 4: COMPARATIVE STATISTICAL ANALYSIS")
                print("=" * 80)
            
            for feature in compare_features:
                compare_feature_between_groups(self.audio_features_df, feature)
        
        # Module 5: Clustering
        clustering_results = None
        if run_clustering:
            self.clustering_analyzer = MusicologicalClusterAnalyzer(self.audio_features_df)
            clustering_results = self.clustering_analyzer.compare_all_methods(auto_optimize=True)
            
            # Save clustering results
            clustering_path = self.config.RESULTS_DIR / 'clustering_comparison.csv'
            clustering_results.to_csv(clustering_path, index=False)
            if FrameworkConfig.VERBOSE:
                print(f"\n✓ Clustering results saved: {clustering_path}")
        
        # Module 6: Classification
        if run_classification and 'label' in self.audio_features_df.columns:
            perform_audio_classification(self.audio_features_df)
        
        # Module 7: Visualization
        if run_clustering and self.clustering_analyzer:
            if FrameworkConfig.VERBOSE:
                print("\n" + "=" * 80)
                print("MODULE 7: DIMENSIONAL VISUALIZATION (PCA)")
                print("=" * 80)
            
            # Visualize by label
            if 'label' in self.audio_features_df.columns:
                save_path = self.config.VISUALIZATIONS_DIR / 'pca_by_label.png'
                visualize_pca_projection(
                    self.audio_features_df.copy(),
                    self.clustering_analyzer.X_scaled,
                    color_column='label',
                    save_path=save_path
                )
            
            # Visualize by cluster (using best method)
            best_method = clustering_results.loc[clustering_results['Silhouette'].idxmax(), 'Method'].lower()
            best_result = self.clustering_analyzer.clustering_results[best_method]
            
            df_with_clusters = self.audio_features_df.copy()
            df_with_clusters['cluster'] = best_result['labels']
            
            save_path = self.config.VISUALIZATIONS_DIR / f'pca_by_{best_method}_clusters.png'
            visualize_pca_projection(
                df_with_clusters,
                self.clustering_analyzer.X_scaled,
                color_column='cluster',
                save_path=save_path
            )
        
        self.results['audio_analysis'] = {
            'features_df': self.audio_features_df,
            'clustering_comparison': clustering_results,
            'n_samples': len(self.audio_features_df)
        }
        
        return self.results['audio_analysis']
    
    # ========================================================================
    # PART 3: SYMBOLIC MUSIC ANALYSIS
    # ========================================================================
    
    def run_symbolic_analysis(self, score_path: str,
                             acknowledge_influences: bool = True,
                             assumed_key: Optional[str] = None) -> Dict:
        """
        Execute symbolic music analysis (Module 8).
        
        Args:
            score_path: Path to MusicXML, MIDI, or other music21-compatible file
            acknowledge_influences: Enable historical attribution
            assumed_key: Optional manual key override (e.g. "C major"), useful
                when automatic key-finding is unreliable (e.g. blues-scale content)
            
        Returns:
            Dictionary with harmonic, contrapuntal, and attribution results
        """
        if not MUSIC21_AVAILABLE:
            if FrameworkConfig.VERBOSE:
                print("\n⚠ music21 not available. Symbolic analysis skipped.")
                print("  Install via: pip install music21")
            return {}
        
        if FrameworkConfig.VERBOSE:
            print("\n" + "▶" * 40)
            print("PART 3: SYMBOLIC MUSIC ANALYSIS")
            print("▶" * 40 + "\n")
        
        self.symbolic_analyzer = HarmonicContrapuntalAnalyzer(
            score_path=score_path,
            acknowledge_influences=acknowledge_influences,
            assumed_key=assumed_key
        )
        
        symbolic_results = self.symbolic_analyzer.generate_comprehensive_report()
        
        self.results['symbolic_analysis'] = symbolic_results
        
        # Save symbolic analysis results
        # (Simplified - full implementation would save DataFrames and metadata)
        output_path = self.config.RESULTS_DIR / 'symbolic_analysis_summary.json'
        summary = {
            'score_name': self.symbolic_analyzer.score_name,
            'key': str(self.symbolic_analyzer.key),
            'n_chords': len(symbolic_results.get('harmony', [])),
            'n_cadences': len(symbolic_results.get('cadences', [])),
            'blue_notes_detected': symbolic_results.get('blue_notes', {}).get('total_occurrences', 0) > 0
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if FrameworkConfig.VERBOSE:
            print(f"\n✓ Symbolic analysis summary saved: {output_path}")
        
        return self.results['symbolic_analysis']
    
    # ========================================================================
    # PART 4: UNIVERSAL INFLUENCE DETECTION
    # ========================================================================
    
    def run_influence_detection(self, analysis_results: Optional[Dict] = None,
                               score_name: str = "Unknown") -> Dict:
        """
        Execute universal influence detection (Module 9).
        
        Args:
            analysis_results: Results from symbolic analysis (Module 8)
                            If None, uses self.results['symbolic_analysis']
            score_name: Name of the analyzed piece
            
        Returns:
            Dictionary with detected influences and attribution report
        """
        if FrameworkConfig.VERBOSE:
            print("\n" + "▶" * 40)
            print("PART 4: UNIVERSAL INFLUENCE DETECTION")
            print("▶" * 40 + "\n")
        
        if analysis_results is None:
            analysis_results = self.results.get('symbolic_analysis', {})
        
        if not analysis_results:
            if FrameworkConfig.VERBOSE:
                print("⚠ No analysis results available for influence detection.")
                print("  Run symbolic analysis first or provide analysis_results parameter.")
            return {}
        
        detected_influences = self.influence_detector.detect_influences(analysis_results)
        
        report = self.influence_detector.generate_influence_report(
            detected_influences,
            score_name
        )
        
        if FrameworkConfig.VERBOSE:
            print(report)
        
        self.results['influence_analysis'] = {
            'detected_influences': detected_influences,
            'report': report,
            'score_name': score_name
        }
        
        # Save influence detection results
        output_path = self.config.RESULTS_DIR / 'influence_detection.json'
        json_safe_results = {
            'score_name': score_name,
            'detected_influences': {
                k: {
                    'confidence': v['confidence'],
                    'interpretation': v['interpretation'],
                    'n_historical_influences': len(v.get('historical_influences', []))
                }
                for k, v in detected_influences.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        if FrameworkConfig.VERBOSE:
            print(f"\n✓ Influence detection results saved: {output_path}")
        
        return self.results['influence_analysis']
    
    # ========================================================================
    # COMPLETE INTEGRATED PIPELINE
    # ========================================================================
    
    def run_complete_pipeline(self,
                             # Synthetic corpus parameters
                             run_synthetic: bool = True,
                             synthetic_corpus_length: int = 1000,
                             
                             # Audio corpus parameters
                             run_audio: bool = True,
                             corpus_directories: Optional[Dict[str, str]] = None,
                             compare_features: Optional[List[str]] = None,
                             
                             # Symbolic analysis parameters
                             run_symbolic: bool = False,
                             score_path: Optional[str] = None,
                             
                             # Influence detection parameters
                             run_influence: bool = False) -> Dict:
        """
        Execute the complete integrated analysis pipeline.
        
        This is the master function that orchestrates all framework modules
        in a coherent workflow.
        
        Args:
            run_synthetic: Execute synthetic corpus analysis (Modules 1-2)
            synthetic_corpus_length: Length of synthetic corpus
            run_audio: Execute audio corpus analysis (Modules 3-7)
            corpus_directories: Dict of {label: directory_path} for audio corpora
            compare_features: List of features to statistically compare
            run_symbolic: Execute symbolic music analysis (Module 8)
            score_path: Path to symbolic music file (MusicXML, MIDI, etc.)
            run_influence: Execute influence detection (Module 9)
            
        Returns:
            Complete results dictionary with all analyses
        """
        
        # ============ PART 1: SYNTHETIC CORPUS ============
        if run_synthetic:
            self.run_synthetic_corpus_analysis(
                corpus_length=synthetic_corpus_length
            )
        
        # ============ PART 2: AUDIO CORPUS ============
        if run_audio:
            if corpus_directories is None:
                corpus_directories = {
                    'classical': str(self.config.CORPUS_DIR / 'classical'),
                    'pop': str(self.config.CORPUS_DIR / 'pop')
                }
            
            if compare_features is None:
                compare_features = ['mfcc_mean_1', 'zcr_mean', 'spectral_centroid_mean']
            
            self.run_audio_corpus_analysis(
                corpus_directories=corpus_directories,
                compare_features=compare_features
            )
        
        # ============ PART 3: SYMBOLIC ANALYSIS ============
        if run_symbolic and score_path:
            self.run_symbolic_analysis(
                score_path=score_path,
                acknowledge_influences=self.config.ENABLE_HISTORICAL_ATTRIBUTION
            )
        
        # ============ PART 4: INFLUENCE DETECTION ============
        if run_influence and self.results.get('symbolic_analysis'):
            score_name = self.symbolic_analyzer.score_name if self.symbolic_analyzer else "Unknown"
            self.run_influence_detection(score_name=score_name)
        
        # ============ FINAL SUMMARY ============
        if FrameworkConfig.VERBOSE:
            self._print_completion_summary()
        
        return self.results
    
    def _print_completion_summary(self):
        """Print final summary of completed analyses."""
        print("\n" + "█" * 80)
        print("█" + " " * 28 + "ANALYSIS COMPLETE" + " " * 35 + "█")
        print("█" * 80)
        
        print("\nCompleted Modules:")
        if self.results.get('synthetic_analysis'):
            print("  ✓ [1-2] Synthetic Corpus Generation & Pattern Analysis")
        if self.results.get('audio_analysis'):
            print("  ✓ [3-7] Audio Feature Extraction, Clustering & Visualization")
        if self.results.get('symbolic_analysis'):
            print("  ✓ [8]   Harmonic & Contrapuntal Analysis")
        if self.results.get('influence_analysis'):
            print("  ✓ [9]   Universal Influence Detection")
        
        print(f"\nResults Directory: {self.config.RESULTS_DIR}")
        print(f"Visualizations Directory: {self.config.VISUALIZATIONS_DIR}")
        
        print("\n" + "█" * 80 + "\n")
    
    def export_complete_results(self, output_filename: str = 'complete_framework_results.json'):
        """
        Export all results to a single JSON file for reproducibility.
        
        Args:
            output_filename: Name of output JSON file
        """
        output_path = self.config.RESULTS_DIR / output_filename
        
        # Create JSON-safe version of results
        export_data = {
            'framework_version': __version__,
            'timestamp': datetime.now().isoformat(),
            'modules_executed': list(self.results.keys()),
            'synthetic_analysis': {
                'executed': bool(self.results.get('synthetic_analysis')),
                'corpus_length': len(self.synthetic_corpus) if self.synthetic_corpus else 0
            },
            'audio_analysis': {
                'executed': bool(self.results.get('audio_analysis')),
                'n_samples': self.results.get('audio_analysis', {}).get('n_samples', 0)
            },
            'symbolic_analysis': {
                'executed': bool(self.results.get('symbolic_analysis'))
            },
            'influence_analysis': {
                'executed': bool(self.results.get('influence_analysis')),
                'n_influences_detected': len(self.results.get('influence_analysis', {}).get('detected_influences', {}))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        if FrameworkConfig.VERBOSE:
            print(f"\n✓ Complete results exported: {output_path}")


# ============================================================================
# CONVENIENCE FUNCTION: Quick Start
# ============================================================================

def quick_start_demo():
    """
    Quick start demonstration of the complete framework.
    
    Runs all modules with default parameters on synthetic data
    and any available audio corpora.
    """
    print(__doc__)
    
    framework = UniversalMusicologicalFramework()
    
    # Configure corpus directories
    corpus_dirs = {
        'classical': './corpus/classical',
        'pop': './corpus/pop'
    }
    
    # Check if directories exist
    audio_available = any(os.path.isdir(d) for d in corpus_dirs.values())
    
    # Run complete pipeline
    results = framework.run_complete_pipeline(
        run_synthetic=True,
        run_audio=audio_available,
        corpus_directories=corpus_dirs if audio_available else None,
        run_symbolic=False,  # Requires MusicXML file
        run_influence=False  # Requires symbolic analysis
    )
    
    # Export results
    framework.export_complete_results()
    
    print("\n" + "─" * 80)
    print("QUICK START DEMO COMPLETE")
    print("─" * 80)
    print("\nTo run symbolic analysis (Module 8) and influence detection (Module 9):")
    print("  framework.run_symbolic_analysis('path/to/score.xml')")
    print("  framework.run_influence_detection()")
    print("\nFor detailed documentation, see:")
    print(f"  {__repository__}")
    print("─" * 80 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    quick_start_demo()