#!/usr/bin/env python3
"""
Quantitative Pattern-Based Framework for Computational Musicological Analysis
==============================================================================

This framework implements a comprehensive Quantitative Pattern-Based (QPB)
methodology for musicological analysis, combining symbolic music representation,
statistical inference, and machine learning techniques.

Author: Benseddik Ahmed
Repository: https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis
Version: 2.0
Licence: MIT

Modules:
1. Synthetic corpus generation (pitch-class representation)
2. Statistical pattern analysis with hypothesis testing
3. Audio feature extraction (MIR descriptors)
4. Comparative statistical analysis
5. Unsupervised clustering (K-means)
6. Supervised classification (Random Forest)
7. Dimensional visualization (PCA)

Citation:
    Benseddik, A. (2025). Quantitative Pattern-Based Framework for 
    Computational Musicological Analysis. GitHub repository.
    https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis

Dependencies:
    numpy>=1.21.0, scipy>=1.7.0, pandas>=1.3.0, librosa>=0.9.0,
    scikit-learn>=1.0.0, matplotlib>=3.4.0
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

__author__ = "Benseddik Ahmed"
__version__ = "2.0"
__repository__ = "https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis"

# ============================================================================
# MODULE 1: Synthetic Musical Corpus Generation (Pitch-Class Representation)
# ============================================================================

class BaroqueCorpusGenerator:
    """
    Generator for synthetic musical corpora based on pitch-class theory.
    
    This class creates controlled musical sequences with intentionally embedded
    motifs, enabling validation of pattern detection algorithms in a controlled
    experimental setting. The implementation follows Forte's (1973) pitch-class
    set theory framework.
    
    Attributes:
        c_major_scale (List[int]): Pitch-class representation of C major scale
        bach_motif (List[int]): BACH motif in German notation (B-A-C-H)
        c_major_arpeggio (List[int]): C major triad arpeggio
        corpus (List[int]): Generated pitch-class sequence
        global_transposition (int): Chromatic transposition offset (0-11)
    
    References:
        Forte, A. (1973). The Structure of Atonal Music. Yale University Press.
    """
    
    def __init__(self, seed: int = 42, global_transposition: int = 0) -> None:
        """
        Initialize the corpus generator with reproducible random state.
        
        Args:
            seed: Random seed for reproducibility
            global_transposition: Chromatic transposition in semitones (mod 12)
        """
        np.random.seed(seed)
        self.c_major_scale = [0, 2, 4, 5, 7, 9, 11]  # C major diatonic scale
        self.bach_motif = [10, 9, 0, 11]  # B♭-A-C-B in German notation
        self.c_major_arpeggio = [0, 4, 7, 0]  # C-E-G-C
        self.corpus = []
        self.global_transposition = global_transposition % 12

    def generate_background_passage(self, length: int) -> List[int]:
        """
        Generate a background musical passage with stylistic variety.
        
        Simulates compositional techniques including scalar motion, arpeggiation,
        repetition, and random selection to create a musically plausible baseline.
        
        Args:
            length: Target length in pitch-class events
            
        Returns:
            List of pitch-class integers representing the generated passage
        """
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

    def insert_motif_intentionally(self, motif: List[int], 
                                   frequency: float) -> int:
        """
        Insert a specific motif at controlled frequency throughout the corpus.
        
        This method implements intentional pattern embedding for experimental
        validation of statistical detection algorithms.
        
        Args:
            motif: Pitch-class sequence to insert
            frequency: Target occurrence frequency (0.0-1.0)
            
        Returns:
            Number of successful motif insertions
        """
        motif_len = len(motif)
        expected_count = int((len(self.corpus) / motif_len) * frequency)
        possible_positions = list(range(0, len(self.corpus) - motif_len + 1, 
                                       motif_len))
        np.random.shuffle(possible_positions)
        positions_to_insert = possible_positions[:expected_count]
        
        for pos in positions_to_insert:
            for i, note in enumerate(motif):
                if pos + i < len(self.corpus):
                    self.corpus[pos + i] = note
                    
        return len(positions_to_insert)

    def transpose_corpus(self) -> None:
        """Apply global chromatic transposition to the entire corpus."""
        self.corpus = [(note + self.global_transposition) % 12 
                      for note in self.corpus]

    def generate_corpus(self, total_length: int = 1000,
                       bach_frequency: float = 0.03,
                       arpeggio_frequency: float = 0.05) -> Tuple[List[int], Dict]:
        """
        Generate a complete synthetic corpus with embedded motifs and metadata.
        
        This is the primary public method for corpus generation. It creates a
        musically plausible baseline sequence and embeds specified motifs at
        controlled frequencies for subsequent statistical analysis.
        
        Args:
            total_length: Target corpus length in pitch-class events
            bach_frequency: Target frequency for BACH motif insertion
            arpeggio_frequency: Target frequency for arpeggio insertion
            
        Returns:
            Tuple containing:
                - corpus (List[int]): Generated pitch-class sequence
                - metadata (Dict): Comprehensive generation statistics
        """
        print("=" * 80)
        print("SYNTHETIC CORPUS GENERATION")
        print("=" * 80)
        
        self.corpus = self.generate_background_passage(total_length // 4)
        bach_count = self.insert_motif_intentionally(self.bach_motif, 
                                                     bach_frequency)
        arpeggio_count = self.insert_motif_intentionally(self.c_major_arpeggio,
                                                         arpeggio_frequency)
        
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

        print(f"Corpus generated: {len(self.corpus)} pitch-class events")
        print(f"BACH motifs inserted: {bach_count}")
        print(f"Arpeggios inserted: {arpeggio_count}")
        print("=" * 80)
        
        return self.corpus, metadata


# ============================================================================
# MODULE 2: Statistical Pattern Analysis with Hypothesis Testing
# ============================================================================

class MusicalPatternAnalyzer:
    """
    Analyzer for musical pattern detection with rigorous statistical validation.
    
    Implements pattern matching algorithms with optional transpositional
    equivalence and applies exact binomial testing to assess statistical
    significance of observed pattern frequencies.
    
    Attributes:
        corpus (List[int]): Pitch-class sequence to analyze
        
    References:
        Conklin, D., & Witten, I. H. (1995). Multiple viewpoint systems for
        music prediction. Journal of New Music Research, 24(1), 51-73.
    """
    
    def __init__(self, corpus: List[int]) -> None:
        """
        Initialize the analyzer with a pitch-class corpus.
        
        Args:
            corpus: List of pitch-class integers (0-11)
        """
        self.corpus = corpus

    def _match_pattern_at(self, corpus: List[int], pattern: List[int], 
                         pos: int, allow_transposition: bool) -> bool:
        """
        Check for pattern match at a specific corpus position.
        
        Supports both exact matching and transpositional equivalence through
        interval-class comparison.
        
        Args:
            corpus: Pitch-class sequence to search
            pattern: Target pattern
            pos: Starting position for comparison
            allow_transposition: Enable transpositional equivalence matching
            
        Returns:
            True if pattern matches at specified position
        """
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
        """
        Exhaustively search for pattern occurrences in the corpus.
        
        Args:
            pattern: Target pitch-class pattern
            allow_transposition: Enable transpositional matching
            
        Returns:
            Dictionary containing occurrence positions and statistics
        """
        positions = [i for i in range(len(self.corpus) - len(pattern) + 1)
                    if self._match_pattern_at(self.corpus, pattern, i, 
                                             allow_transposition)]
        frequency = len(positions) / len(self.corpus) if self.corpus else 0
        
        return {
            'pattern': pattern,
            'occurrences': len(positions),
            'positions': positions,
            'frequency': frequency
        }

    def statistical_test_binomial(self, observed: int, corpus_size: int,
                                  expected_prob: float) -> Dict:
        """
        Perform exact binomial test for pattern over-representation.
        
        Tests the null hypothesis that the pattern occurs at random with
        probability expected_prob against the alternative hypothesis of
        over-representation (one-tailed test).
        
        Args:
            observed: Number of observed pattern occurrences
            corpus_size: Total number of possible pattern positions
            expected_prob: Expected probability under null hypothesis
            
        Returns:
            Dictionary with test statistics and interpretation
            
        References:
            Agresti, A. (2018). Statistical Methods for the Social Sciences.
            Pearson Education.
        """
        result = stats.binomtest(observed, corpus_size, expected_prob, 
                                alternative='greater')
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

    def analyze_pattern(self, pattern: List[int], 
                       pattern_name: str) -> Dict:
        """
        Conduct comprehensive analysis of a musical pattern.
        
        Combines pattern detection with statistical significance testing and
        provides musicologically interpretable output.
        
        Args:
            pattern: Pitch-class pattern to analyze
            pattern_name: Descriptive label for reporting
            
        Returns:
            Dictionary containing full analysis results
        """
        print(f"\n{'=' * 70}")
        print(f"Pattern Analysis: {pattern_name}")
        print('=' * 70)
        
        occ = self.find_pattern_occurrences(pattern)
        print(f"Occurrences: {occ['occurrences']}")
        print(f"Observed frequency: {occ['frequency']:.6f}")

        expected_prob = 1.0 / (len(set(self.corpus)) ** len(pattern))
        binom_res = self.statistical_test_binomial(
            occ['occurrences'], 
            len(self.corpus) - len(pattern) + 1, 
            expected_prob
        )
        
        print(f"Enrichment ratio: {binom_res['enrichment_ratio']:.2f}×")
        print(f"P-value: {binom_res['p_value']:.6f}")
        print(f"Significant (α=0.05): {binom_res['significant']}")

        interpretation = (
            "✓ Intentional motif detected" 
            if binom_res['significant'] 
            else "✗ No significant pattern"
        )
        print(f"Interpretation: {interpretation}")
        
        return {
            'pattern_name': pattern_name,
            'occurrences': occ,
            'binomial_test': binom_res
        }


# ============================================================================
# MODULE 3: Advanced Audio Feature Extraction
# ============================================================================

def extract_audio_features(file_path: str, n_mfcc: int = 13, 
                          duration: Optional[float] = 30) -> Dict[str, float]:
    """
    Extract comprehensive Music Information Retrieval (MIR) features from audio.
    
    Computes timbral, spectral, and temporal features following established
    MIR conventions. All features are aggregated using mean and variance to
    provide robust statistical descriptors.
    
    Args:
        file_path: Path to audio file (.wav format recommended)
        n_mfcc: Number of Mel-Frequency Cepstral Coefficients to extract
        duration: Maximum audio duration to process (seconds)
        
    Returns:
        Dictionary of feature names and values
        
    Features:
        - MFCC: Timbral envelope (mean + variance per coefficient)
        - ZCR: Zero-crossing rate (noisy/harmonic content indicator)
        - Spectral Centroid: Spectral brightness measure
        - Spectral Roll-off: High-frequency energy distribution
        
    References:
        Peeters, G. (2004). A large set of audio features for sound description.
        CUIDADO Project Report.
        
        McFee, B., et al. (2015). librosa: Audio and music signal analysis in
        Python. SciPy.
    """
    y, sr = librosa.load(file_path, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    features = {}
    
    # MFCC statistics (timbral descriptors)
    for i in range(n_mfcc):
        features[f'mfcc_mean_{i+1}'] = float(np.mean(mfccs[i]))
        features[f'mfcc_var_{i+1}'] = float(np.var(mfccs[i]))
    
    # Zero-crossing rate (temporal descriptor)
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_var'] = float(np.var(zcr))
    
    # Spectral centroid (brightness descriptor)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
    features['spectral_centroid_var'] = float(np.var(spectral_centroid))
    
    # Spectral roll-off (high-frequency content descriptor)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = float(np.mean(rolloff))
    features['rolloff_var'] = float(np.var(rolloff))
    
    return features


def load_corpus_audio_features(corpus_path: str, 
                               label: Optional[str] = None) -> pd.DataFrame:
    """
    Extract features from all audio files in a corpus directory.
    
    Args:
        corpus_path: Directory containing .wav audio files
        label: Optional categorical label for supervised learning
        
    Returns:
        DataFrame with one row per audio file and columns for each feature
    """
    feature_list = []
    audio_files = [f for f in os.listdir(corpus_path) if f.endswith('.wav')]
    
    for filename in audio_files:
        filepath = os.path.join(corpus_path, filename)
        features = extract_audio_features(filepath)
        features['filename'] = filename
        if label:
            features['label'] = label
        feature_list.append(features)
    
    return pd.DataFrame(feature_list)


# ============================================================================
# MODULE 4: Comparative Statistical Analysis
# ============================================================================

def compare_feature_between_groups(df: pd.DataFrame, 
                                  feature_name: str) -> None:
    """
    Perform independent samples t-test between two musical groups.
    
    Implements Welch's t-test (unequal variances assumed) to compare
    distributions of a specific audio feature across musical categories.
    Appropriate for testing hypotheses about stylistic differentiation.
    
    Args:
        df: DataFrame with 'label' column and target feature
        feature_name: Name of feature column to compare
        
    Statistical Test:
        H0: μ₁ = μ₂ (no difference in population means)
        H1: μ₁ ≠ μ₂ (significant difference exists)
        
    References:
        Welch, B. L. (1947). The generalization of "Student's" problem when
        several different population variances are involved. Biometrika, 34(1-2).
    """
    group_labels = df['label'].unique()
    if len(group_labels) != 2:
        print("⚠ Function optimized for binary comparison only.")
        return
    
    grp1 = df[df['label'] == group_labels[0]][feature_name]
    grp2 = df[df['label'] == group_labels[1]][feature_name]
    
    t_stat, p_val = stats.ttest_ind(grp1, grp2, equal_var=False)
    
    print(f"\n--- Comparative Analysis: '{feature_name}' ---")
    print(f"Group 1 ({group_labels[0]}): μ={grp1.mean():.4f}, σ={grp1.std():.4f}")
    print(f"Group 2 ({group_labels[1]}): μ={grp2.mean():.4f}, σ={grp2.std():.4f}")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4g}")
    print(f"Significant difference (α=0.05): {'Yes' if p_val < 0.05 else 'No'}")


# ============================================================================
# MODULE 5: Unsupervised Clustering (K-means)
# ============================================================================

def perform_audio_clustering(df: pd.DataFrame, 
                            n_clusters: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Discover latent groupings in audio feature space using K-means clustering.
    
    Applies standardization (z-score normalization) prior to clustering to
    prevent scale-dependent bias. Useful for exploratory analysis and
    validation of known taxonomies.
    
    Args:
        df: DataFrame with audio features
        n_clusters: Number of clusters (k parameter)
        
    Returns:
        Tuple containing:
            - Modified DataFrame with 'cluster' column
            - Standardized feature matrix for visualization
            
    Notes:
        Optimal k can be determined via elbow method or silhouette analysis.
        
    References:
        MacQueen, J. (1967). Some methods for classification and analysis of
        multivariate observations. Proceedings of the Fifth Berkeley Symposium.
    """
    X = df.drop(columns=['filename', 'label'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    return df, X_scaled


# ============================================================================
# MODULE 6: Supervised Classification (Random Forest)
# ============================================================================

def perform_audio_classification(df: pd.DataFrame) -> None:
    """
    Train and evaluate a Random Forest classifier for musical style recognition.
    
    Assesses the discriminative power of extracted audio features for
    predicting musical categories. Provides comprehensive performance metrics
    including precision, recall, F1-score, and support statistics.
    
    Args:
        df: DataFrame with 'label' column (ground truth) and feature columns
        
    Model Configuration:
        - Algorithm: Random Forest (ensemble of 100 decision trees)
        - Training split: 70% train / 30% test (stratified)
        - Preprocessing: Z-score standardization
        
    Output:
        Prints classification report with per-class and aggregate metrics
        
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
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Random Forest)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, digits=3))


# ============================================================================
# MODULE 7: Dimensional Visualization (PCA)
# ============================================================================

def visualize_pca_projection(df: pd.DataFrame, X_scaled: np.ndarray,
                            color_column: str = 'label') -> None:
    """
    Generate 2D PCA projection for visual exploration of feature space structure.
    
    Principal Component Analysis reduces high-dimensional audio features to
    two dimensions while maximizing variance retention. Useful for assessing
    cluster coherence and inter-class separability.
    
    Args:
        df: DataFrame with categorical variable for coloration
        X_scaled: Standardized feature matrix
        color_column: Column name for group assignment ('label' or 'cluster')
        
    Visualization:
        - Scatter plot with group-based coloration
        - Axis labels show variance explained by each component
        - Legend indicates group membership
        
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
    plt.xlabel(f'Principal Component 1 ({var_explained[0]:.1%} variance)', 
               fontsize=12)
    plt.ylabel(f'Principal Component 2 ({var_explained[1]:.1%} variance)', 
               fontsize=12)
    plt.title(f'PCA Projection (colored by {color_column})', 
             fontsize=14, fontweight='bold')
    plt.legend(title=color_column.capitalize(), fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


# ============================================================================
# INTEGRATED PIPELINE: Complete Demonstration Workflow
# ============================================================================

def run_complete_analysis_pipeline() -> None:
    """
    Execute the complete QPB musicological analysis workflow.
    
    This function orchestrates all framework modules to demonstrate:
    1. Synthetic corpus generation and validation
    2. Statistical pattern detection with hypothesis testing
    3. Real audio corpus feature extraction
    4. Comparative statistical analysis
    5. Unsupervised clustering
    6. Supervised classification
    7. Visual exploration via PCA
    
    Workflow Design:
        Part 1: Controlled experiment on synthetic data (method validation)
        Part 2: Application to real-world audio corpus (empirical analysis)
        
    Output:
        Comprehensive console reports, statistical summaries, and visualizations
    """
    
    print("\n" + "█" * 80)
    print("█" + " " * 15 + "QUANTITATIVE PATTERN-BASED FRAMEWORK v2.0" + " " * 23 + "█")
    print("█" + " " * 25 + "Author: Benseddik Ahmed" + " " * 32 + "█")
    print("█" * 80 + "\n")
    
    # ========== PART 1: SYNTHETIC CORPUS ANALYSIS ==========
    print("\n▶ PART 1: SYNTHETIC CORPUS GENERATION AND VALIDATION\n")
    
    generator = BaroqueCorpusGenerator(seed=42)
    corpus, metadata = generator.generate_corpus(
        total_length=1000,
        bach_frequency=0.03,
        arpeggio_frequency=0.05
    )
    
    analyzer = MusicalPatternAnalyzer(corpus)
    
    test_patterns = {
        'BACH Motif (B-A-C-H)': [10, 9, 0, 11],
        'C Major Arpeggio': [0, 4, 7, 0],
        'Ascending Scale Fragment': [0, 2, 4, 5],
        'Random Control Pattern': [3, 8, 1, 6]
    }
    
    results_synthetic = {}
    for name, pattern in test_patterns.items():
        results_synthetic[name] = analyzer.analyze_pattern(pattern, name)
    
    # ========== PART 2: REAL AUDIO CORPUS ANALYSIS ==========
    print("\n\n▶ PART 2: REAL AUDIO CORPUS EXTRACTION AND ANALYSIS\n")
    
    corpus_directories = {
        'classical': './corpus/classical',
        'pop': './corpus/pop'
    }
    
    dataframes = []
    for label, directory in corpus_directories.items():
        if not os.path.isdir(directory):
            print(f"⚠ Directory '{directory}' not found. Skipping '{label}' corpus.")
            print(f"  To enable analysis, create directory and add .wav files:")
            print(f"  $ mkdir -p {directory}")
            continue
        
        print(f"Extracting features for '{label}' corpus...")
        df_subset = load_corpus_audio_features(directory, label=label)
        dataframes.append(df_subset)
    
    if not dataframes:
        print("\n⚠ NO AUDIO CORPUS DETECTED")
        print("To utilize audio analysis capabilities:")
        for label, path in corpus_directories.items():
            print(f"  1. Create directory: {path}")
            print(f"  2. Add .wav files of {label} music")
        print("\nFramework demonstration completed (synthetic analysis only).")
        return
    
    df_audio = pd.concat(dataframes, ignore_index=True)
    print(f"\n✓ Total audio excerpts analyzed: {len(df_audio)}")
    
    # Statistical comparative analysis
    print("\n" + "─" * 70)
    print("COMPARATIVE STATISTICAL ANALYSIS")
    print("─" * 70)
    compare_feature_between_groups(df_audio, 'mfcc_mean_1')
    compare_feature_between_groups(df_audio, 'zcr_mean')
    compare_feature_between_groups(df_audio, 'spectral_centroid_mean')
    
    # Unsupervised clustering
    print("\n" + "─" * 70)
    print("UNSUPERVISED CLUSTERING (K-MEANS)")
    print("─" * 70)
    df_audio, X_scaled = perform_audio_clustering(df_audio, n_clusters=2)
    print("\nCluster assignments (sample):")
    print(df_audio[['filename', 'label', 'cluster']].head(10))
    
    # Supervised classification
    print("\n" + "─" * 70)
    print("SUPERVISED CLASSIFICATION")
    print("─" * 70)
    perform_audio_classification(df_audio)
    
    # Visual exploration via PCA
    print("\n" + "─" * 70)
    print("DIMENSIONAL VISUALIZATION (PCA)")
    print("─" * 70)
    print("Generating visualizations...")
    
    visualize_pca_projection(df_audio, X_scaled, color_column='label')
    visualize_pca_projection(df_audio, X_scaled, color_column='cluster')
    
    print("\n" + "█" * 80)
    print("█" + " " * 28 + "ANALYSIS COMPLETE" + " " * 35 + "█")
    print("█" * 80 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print(f"""
{'='*80}
Quantitative Pattern-Based Framework for Computational Musicology
Author: {__author__}
Version: {__version__}
Repository: {__repository__}
{'='*80}
""")
    run_complete_analysis_pipeline()