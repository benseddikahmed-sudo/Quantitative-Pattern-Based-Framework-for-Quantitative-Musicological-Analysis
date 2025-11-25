# ============================================================================
# MODULE 5: Advanced Unsupervised Clustering with Hyperparameter Optimization
# ============================================================================

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠ hdbscan not installed. Install via: pip install hdbscan")


class MusicologicalClusterAnalyzer:
    """
    Advanced clustering system for musicological corpus analysis.
    
    Integrates multiple clustering algorithms with automatic hyperparameter
    optimization and musicologically-informed validation metrics.
    
    Attributes:
        df (pd.DataFrame): Audio feature DataFrame
        X_scaled (np.ndarray): Standardized feature matrix
        scaler (StandardScaler): Fitted scaler for inverse transformations
        clustering_results (Dict): Storage for all clustering outcomes
        
    References:
        Ester, M., et al. (1996). A density-based algorithm for discovering
        clusters in large spatial databases with noise. KDD-96.
        
        Campello, R. J., et al. (2013). Density-based clustering based on
        hierarchical density estimates. PAKDD.
        
        McInnes, L., et al. (2017). hdbscan: Hierarchical density based
        clustering. Journal of Open Source Software, 2(11), 205.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the clustering analyzer.
        
        Args:
            df: DataFrame with audio features and optional 'label' column
        """
        self.df = df.copy()
        self.X = df.drop(columns=['filename', 'label', 'cluster', 
                                  'cluster_probability'], errors='ignore')
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.clustering_results = {}
        
        print(f"✓ Initialized analyzer with {len(df)} samples, "
              f"{self.X.shape[1]} features")
    
    # ========================================================================
    # HYPERPARAMETER OPTIMIZATION METHODS
    # ========================================================================
    
    def optimize_dbscan_eps(self, k: int = 4, 
                           percentile_range: Tuple[int, int] = (90, 99)) -> float:
        """
        Automatically determine optimal epsilon for DBSCAN using k-distance graph.
        
        The k-distance graph method identifies the "elbow" point where the
        distance to the k-th nearest neighbor increases sharply, indicating
        the transition from dense to sparse regions.
        
        Args:
            k: Number of nearest neighbors (typically min_samples - 1)
            percentile_range: Range for elbow detection (start, end percentiles)
            
        Returns:
            Optimal epsilon value
            
        Methodology:
            1. Compute k-nearest neighbor distances for all points
            2. Sort distances in ascending order
            3. Identify elbow point using percentile heuristic
            4. Return distance at elbow as epsilon
            
        References:
            Schubert, E., et al. (2017). DBSCAN revisited: Why and how you
            should (still) use DBSCAN. ACM TODS, 42(3), 1-21.
        """
        print("\n→ Optimizing DBSCAN epsilon parameter...")
        
        # Compute k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, _ = neighbors.kneighbors(self.X_scaled)
        
        # Sort k-th nearest neighbor distances
        k_distances = np.sort(distances[:, k-1])
        
        # Elbow detection using percentile method
        n_samples = len(k_distances)
        start_idx = int(n_samples * percentile_range[0] / 100)
        end_idx = int(n_samples * percentile_range[1] / 100)
        
        # Find maximum curvature point
        gradients = np.gradient(k_distances[start_idx:end_idx])
        elbow_idx = start_idx + np.argmax(gradients)
        optimal_eps = k_distances[elbow_idx]
        
        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(k_distances, linewidth=2, color='steelblue')
        plt.axhline(y=optimal_eps, color='red', linestyle='--', 
                   label=f'Optimal ε = {optimal_eps:.3f}')
        plt.axvline(x=elbow_idx, color='orange', linestyle='--', alpha=0.5,
                   label=f'Elbow at sample {elbow_idx}')
        plt.xlabel('Sample Index (sorted by distance)', fontsize=12)
        plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
        plt.title('K-Distance Graph for DBSCAN ε Optimization', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"  ✓ Optimal ε: {optimal_eps:.4f}")
        return optimal_eps
    
    def optimize_dbscan_min_samples(self, eps: float,
                                    range_multiplier: Tuple[int, int] = (2, 10)) -> int:
        """
        Determine optimal min_samples for DBSCAN by maximizing silhouette score.
        
        Tests multiple min_samples values and selects the one producing the
        highest clustering quality according to silhouette coefficient.
        
        Args:
            eps: Epsilon value (from optimize_dbscan_eps)
            range_multiplier: (min, max) multipliers of feature dimensionality
            
        Returns:
            Optimal min_samples value
            
        Notes:
            Rule of thumb: min_samples ≥ dimensionality + 1
            For high-dimensional data: min_samples = 2 × dimensionality
        """
        print("\n→ Optimizing DBSCAN min_samples parameter...")
        
        dim = self.X_scaled.shape[1]
        min_range = max(2, dim * range_multiplier[0] // 10)
        max_range = min(dim * range_multiplier[1] // 10, len(self.df) // 4)
        
        candidates = range(min_range, max_range + 1)
        scores = []
        
        for min_samp in candidates:
            dbscan = DBSCAN(eps=eps, min_samples=min_samp)
            labels = dbscan.fit_predict(self.X_scaled)
            
            # Compute silhouette (excluding noise)
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(self.X_scaled, labels)
            elif len(set(labels)) > 2:  # Has noise but multiple clusters
                mask = labels != -1
                if mask.sum() > 1:
                    score = silhouette_score(self.X_scaled[mask], labels[mask])
                else:
                    score = -1
            else:
                score = -1
            
            scores.append(score)
        
        # Select best min_samples
        valid_scores = [(s, m) for s, m in zip(scores, candidates) if s > 0]
        if valid_scores:
            optimal_min_samples = max(valid_scores, key=lambda x: x[0])[1]
        else:
            optimal_min_samples = min_range
        
        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(list(candidates), scores, marker='o', linewidth=2, 
                markersize=6, color='steelblue')
        plt.axvline(x=optimal_min_samples, color='red', linestyle='--',
                   label=f'Optimal min_samples = {optimal_min_samples}')
        plt.xlabel('min_samples', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('DBSCAN min_samples Optimization', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"  ✓ Optimal min_samples: {optimal_min_samples}")
        return optimal_min_samples
    
    def optimize_hdbscan_parameters(self) -> Dict[str, int]:
        """
        Optimize HDBSCAN hyperparameters using DBSCAN-Validity Index (DBCV).
        
        HDBSCAN is less sensitive to parameters than DBSCAN, but optimal
        min_cluster_size and min_samples still improve results significantly.
        
        Returns:
            Dictionary with optimal 'min_cluster_size' and 'min_samples'
            
        Strategy:
            - min_cluster_size: Test range from 2% to 10% of dataset size
            - min_samples: Test range from 1 to min_cluster_size
            - Optimize using DBCV (built-in HDBSCAN validity metric)
        """
        if not HDBSCAN_AVAILABLE:
            return {'min_cluster_size': 5, 'min_samples': 3}
        
        print("\n→ Optimizing HDBSCAN parameters...")
        
        n_samples = len(self.df)
        min_cluster_sizes = [int(n_samples * p) for p in [0.02, 0.03, 0.05, 0.07, 0.10]]
        min_cluster_sizes = [max(2, s) for s in min_cluster_sizes]
        
        best_score = -np.inf
        best_params = {'min_cluster_size': 5, 'min_samples': 3}
        
        results = []
        
        for mcs in min_cluster_sizes:
            for ms in range(1, min(mcs, 10)):
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric='euclidean'
                    )
                    labels = clusterer.fit_predict(self.X_scaled)
                    
                    # Use validity score if available
                    if hasattr(clusterer, 'relative_validity_'):
                        score = clusterer.relative_validity_
                    else:
                        # Fallback to silhouette
                        mask = labels != -1
                        if mask.sum() > 1 and len(set(labels[mask])) > 1:
                            score = silhouette_score(self.X_scaled[mask], labels[mask])
                        else:
                            score = -1
                    
                    results.append({
                        'min_cluster_size': mcs,
                        'min_samples': ms,
                        'score': score,
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'min_cluster_size': mcs, 'min_samples': ms}
                
                except Exception as e:
                    continue
        
        # Visualization
        df_results = pd.DataFrame(results)
        if len(df_results) > 0:
            pivot = df_results.pivot_table(
                values='score', 
                index='min_samples', 
                columns='min_cluster_size',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            plt.imshow(pivot, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Validity Score')
            plt.xlabel('min_cluster_size', fontsize=12)
            plt.ylabel('min_samples', fontsize=12)
            plt.title('HDBSCAN Parameter Optimization Heatmap',
                     fontsize=14, fontweight='bold')
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.tight_layout()
            plt.show()
        
        print(f"  ✓ Optimal min_cluster_size: {best_params['min_cluster_size']}")
        print(f"  ✓ Optimal min_samples: {best_params['min_samples']}")
        print(f"  ✓ Best validity score: {best_score:.4f}")
        
        return best_params
    
    # ========================================================================
    # CLUSTERING EXECUTION METHODS
    # ========================================================================
    
    def run_kmeans(self, n_clusters: int = 3) -> Dict:
        """
        Execute K-means clustering with evaluation metrics.
        
        Args:
            n_clusters: Number of clusters to generate
            
        Returns:
            Dictionary with labels, metrics, and model object
        """
        print(f"\n→ Running K-means (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, 
                       n_init=20, max_iter=300)
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
    
    def run_dbscan(self, eps: Optional[float] = None, 
                   min_samples: Optional[int] = None,
                   auto_optimize: bool = True) -> Dict:
        """
        Execute DBSCAN clustering with optional automatic optimization.
        
        Args:
            eps: Epsilon parameter (auto-optimized if None and auto_optimize=True)
            min_samples: Minimum samples (auto-optimized if None)
            auto_optimize: Enable automatic hyperparameter optimization
            
        Returns:
            Dictionary with labels, metrics, parameters, and model object
        """
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
    
    def run_hdbscan(self, min_cluster_size: Optional[int] = None,
                    min_samples: Optional[int] = None,
                    auto_optimize: bool = True) -> Dict:
        """
        Execute HDBSCAN clustering with optional automatic optimization.
        
        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core points
            auto_optimize: Enable automatic hyperparameter optimization
            
        Returns:
            Dictionary with labels, metrics, parameters, probabilities, and model
        """
        if not HDBSCAN_AVAILABLE:
            print("⚠ HDBSCAN not available, skipping...")
            return {}
        
        print(f"\n→ Running HDBSCAN clustering...")
        
        if auto_optimize and (min_cluster_size is None or min_samples is None):
            optimal_params = self.optimize_hdbscan_parameters()
            min_cluster_size = optimal_params['min_cluster_size']
            min_samples = optimal_params['min_samples']
        else:
            if min_cluster_size is None:
                min_cluster_size = max(5, len(self.df) // 20)
            if min_samples is None:
                min_samples = 3
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            gen_min_span_tree=True
        )
        labels = clusterer.fit_predict(self.X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = self._compute_clustering_metrics(labels, 'HDBSCAN')
        
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
    
    # ========================================================================
    # EVALUATION AND VALIDATION METHODS
    # ========================================================================
    
    def _compute_clustering_metrics(self, labels: np.ndarray, 
                                    method_name: str) -> Dict[str, float]:
        """
        Compute comprehensive clustering quality metrics.
        
        Args:
            labels: Cluster assignments
            method_name: Algorithm name for reporting
            
        Returns:
            Dictionary of metric names and values
            
        Metrics:
            - Silhouette Coefficient: Measures cluster cohesion and separation
            - Davies-Bouldin Index: Lower is better (avg. cluster similarity)
            - Calinski-Harabasz Index: Higher is better (variance ratio)
        """
        # Filter out noise points for metric computation
        mask = labels != -1
        valid_labels = labels[mask]
        valid_data = self.X_scaled[mask]
        
        metrics = {}
        
        if len(set(valid_labels)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(valid_data, valid_labels)
                metrics['davies_bouldin'] = davies_bouldin_score(valid_data, valid_labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_labels)
            except Exception as e:
                print(f"  ⚠ Could not compute some metrics: {e}")
                metrics['silhouette'] = -1
                metrics['davies_bouldin'] = -1
                metrics['calinski_harabasz'] = -1
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        return metrics
    
    def compare_all_methods(self, auto_optimize: bool = True) -> pd.DataFrame:
        """
        Execute all clustering methods and generate comparative report.
        
        Args:
            auto_optimize: Enable automatic hyperparameter tuning
            
        Returns:
            DataFrame with comparative metrics for all methods
        """
        print("\n" + "=" * 80)
        print("COMPARATIVE CLUSTERING ANALYSIS")
        print("=" * 80)
        
        # Run all methods
        self.run_kmeans(n_clusters=3)
        self.run_dbscan(auto_optimize=auto_optimize)
        self.run_hdbscan(auto_optimize=auto_optimize)
        
        # Compile results
        comparison = []
        for method, result in self.clustering_results.items():
            if result:  # Skip empty results (e.g., HDBSCAN if not available)
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
        
        print("\n" + "─" * 80)
        print("QUANTITATIVE COMPARISON")
        print("─" * 80)
        print(df_comparison.to_string(index=False))
        print("\nMetric Interpretation:")
        print("  • Silhouette ∈ [-1, 1]: Higher is better (>0.5 excellent)")
        print("  • Davies-Bouldin: Lower is better (<1.0 good)")
        print("  • Calinski-Harabasz: Higher is better (>100 good)")
        
        return df_comparison
    
    def musicological_interpretation(self, method: str = 'hdbscan') -> None:
        """
        Provide musicological interpretation of clustering results.
        
        Args:
            method: Clustering method to interpret ('kmeans', 'dbscan', 'hdbscan')
        """
        if method not in self.clustering_results:
            print(f"⚠ No results available for {method}")
            return
        
        result = self.clustering_results[method]
        labels = result['labels']
        
        print("\n" + "=" * 80)
        print(f"MUSICOLOGICAL INTERPRETATION - {method.upper()}")
        print("=" * 80)
        
        # Cluster size distribution
        unique_labels = set(labels)
        cluster_sizes = {lbl: list(labels).count(lbl) for lbl in unique_labels}
        
        print("\nCluster Size Distribution:")
        for lbl in sorted(cluster_sizes.keys()):
            if lbl == -1:
                print(f"  Noise/Outliers: {cluster_sizes[lbl]} samples "
                      f"({cluster_sizes[lbl]/len(labels)*100:.1f}%)")
                print("    → Potential musical innovations or transitional passages")
            else:
                print(f"  Cluster {lbl}: {cluster_sizes[lbl]} samples "
                      f"({cluster_sizes[lbl]/len(labels)*100:.1f}%)")
        
        # Feature analysis per cluster
        if 'label' in self.df.columns:
            print("\nGround Truth vs. Clustering:")
            df_temp = self.df.copy()
            df_temp['cluster'] = labels
            
            contingency = pd.crosstab(
                df_temp['label'], 
                df_temp['cluster'],
                margins=True
            )
            print(contingency)
            
            # Purity calculation
            if len(unique_labels) > 1:
                purity = sum(contingency.max(axis=0)[:-1]) / contingency.loc['All', 'All']
                print(f"\nClustering Purity: {purity:.3f}")
                print("  → Proportion of samples in their majority-label cluster")
        
        # Cluster feature centroids
        print("\nCluster Acoustic Profiles (Top 3 Discriminative Features):")
        df_temp = self.df.copy()
        df_temp['cluster'] = labels
        
        for lbl in sorted([l for l in unique_labels if l != -1]):
            cluster_data = df_temp[df_temp['cluster'] == lbl]
            cluster_features = cluster_data[self.X.columns].mean()
            
            # Compute feature importance (distance from global mean)
            global_mean = self.df[self.X.columns].mean()
            deviations = abs(cluster_features - global_mean) / global_mean.std()
            top_features = deviations.nlargest(3)
            
            print(f"\n  Cluster {lbl}:")
            for feat, dev in top_features.items():
                direction = "↑ High" if cluster_features[feat] > global_mean[feat] else "↓ Low"
                print(f"    • {feat}: {direction} ({dev:.2f}σ)")
    
    def visualize_clustering_results(self) -> None:
        """
        Generate comprehensive visualizations for all clustering methods.
        
        Creates:
            1. PCA projections for each method
            2. Silhouette plots for quality assessment
            3. Cluster size distributions
        """
        n_methods = len(self.clustering_results)
        if n_methods == 0:
            print("⚠ No clustering results to visualize")
            return
        
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(self.X_scaled)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, result) in enumerate(self.clustering_results.items()):
            if not result:
                continue
            
            ax = axes[idx]
            labels = result['labels']
            unique_labels = sorted(set(labels))
            
            # Color palette
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                if lbl == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c='gray', marker='x', s=50, alpha=0.5,
                             label='Noise', edgecolors='k', linewidths=0.5)
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                             c=[colors[i]], marker='o', s=100, alpha=0.7,
                             label=f'Cluster {lbl}', edgecolors='k', linewidths=0.5)
            
            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)', fontsize=12)
            
            title = f"{method.upper()}\n"
            title += f"Silhouette: {result['metrics']['silhouette']:.3f}"
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, output_path: str = './clustering_results.json') -> None:
        """
        Export clustering results to JSON for reproducibility.
        
        Args:
            output_path: Path for output JSON file
        """
        export_data = {}
        
        for method, result in self.clustering_results.items():
            if not result:
                continue
            
            export_data[method] = {
                'n_clusters': int(result['n_clusters']),
                'labels': result['labels'].tolist(),
                'metrics': {k: float(v) for k, v in result['metrics'].items()},
            }
            
            if 'eps' in result:
                export_data[method]['eps'] = float(result['eps'])
            if 'min_samples' in result:
                export_data[method]['min_samples'] = int(result['min_samples'])
            if 'min_cluster_size' in result:
                export_data[method]['min_cluster_size'] = int(result['min_cluster_size'])
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Results exported to: {output_path}")


# ============================================================================
# UPDATED MODULE 5 WRAPPER FUNCTIONS (for backward compatibility)
# ============================================================================

def perform_audio_clustering(df: pd.DataFrame, 
                            method: str = 'kmeans',
                            n_clusters: int = 3,
                            auto_optimize: bool = True,
                            **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simplified wrapper for backward compatibility with original interface.
    
    Args:
        df: DataFrame with audio features
        method: 'kmeans', 'dbscan', or 'hdbscan'
        n_clusters: For K-means only
        auto_optimize: Enable automatic hyperparameter tuning
        **kwargs: Additional parameters passed to specific methods
        
    Returns:
        Tuple of (DataFrame with cluster column, standardized features)
    """
    analyzer = MusicologicalClusterAnalyzer(df)
    
    if method == 'kmeans':
        result = analyzer.run_kmeans(n_clusters=n_clusters)
    elif method == 'dbscan':
        result = analyzer.run_dbscan(auto_optimize=auto_optimize, **kwargs)
    elif method == 'hdbscan':
        result = analyzer.run_hdbscan(auto_optimize=auto_optimize, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df_out = df.copy()
    df_out['cluster'] = result['labels']
    
    if 'probabilities' in result and result['probabilities'] is not None:
        df_out['cluster_probability'] = result['probabilities']
    
    return df_out, analyzer.X_scaled
