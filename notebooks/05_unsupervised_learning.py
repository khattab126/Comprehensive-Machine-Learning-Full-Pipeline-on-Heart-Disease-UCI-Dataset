"""
Unsupervised Learning for Heart Disease Dataset
K-Means Clustering and Hierarchical Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed heart disease data"""
    try:
        df = pd.read_csv('../data/heart_disease_selected_features.csv')
        print(f"Selected features data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        try:
            df = pd.read_csv('../data/heart_disease_processed.csv')
            # Use all features except target and dataset
            feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
            df = df[feature_cols + ['target', 'dataset']]
            print(f"Full processed data loaded. Shape: {df.shape}")
        except FileNotFoundError:
            print("Processed data not found. Please run data preprocessing first.")
            return None
    
    return df

def prepare_data(df):
    """Prepare data for clustering"""
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    X = df[feature_cols]
    y = df['target']
    datasets = df['dataset'] if 'dataset' in df.columns else None
    
    print(f"Features for clustering: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, datasets, feature_cols, scaler

def find_optimal_k(X_scaled, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette analysis"""
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    # Calculate metrics for different k values
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal K based on silhouette score: {optimal_k}")
    
    # Plot elbow and silhouette curves
    plt.figure(figsize=(15, 5))
    
    # Elbow curve
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # Silhouette curve
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    
    # Combined plot
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax1.plot(k_range, inertias, 'bo-', label='Inertia')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Elbow Method vs Silhouette Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_k, inertias, silhouette_scores

def kmeans_clustering(X_scaled, optimal_k, y_true=None):
    """Perform K-Means clustering"""
    print(f"\n" + "="*60)
    print(f"K-MEANS CLUSTERING (K={optimal_k})")
    print("="*60)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.3f}")
    
    # Compare with true labels if available
    if y_true is not None:
        ari = adjusted_rand_score(y_true, cluster_labels)
        nmi = normalized_mutual_info_score(y_true, cluster_labels)
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Normalized Mutual Info: {nmi:.3f}")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_dist = dict(zip(unique, counts))
    print(f"Cluster distribution: {cluster_dist}")
    
    return kmeans, cluster_labels, silhouette

def hierarchical_clustering(X_scaled, optimal_k, y_true=None):
    """Perform Hierarchical clustering"""
    print(f"\n" + "="*60)
    print(f"HIERARCHICAL CLUSTERING (K={optimal_k})")
    print("="*60)
    
    # Different linkage methods
    linkage_methods = ['ward', 'complete', 'average', 'single']
    best_method = 'ward'
    best_silhouette = -1
    
    results = {}
    
    for method in linkage_methods:
        print(f"\nTesting {method} linkage...")
        
        # Fit hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, cluster_labels)
        print(f"  Silhouette Score: {silhouette:.3f}")
        
        # Compare with true labels if available
        if y_true is not None:
            ari = adjusted_rand_score(y_true, cluster_labels)
            nmi = normalized_mutual_info_score(y_true, cluster_labels)
            print(f"  Adjusted Rand Index: {ari:.3f}")
            print(f"  Normalized Mutual Info: {nmi:.3f}")
        
        results[method] = {
            'model': hierarchical,
            'labels': cluster_labels,
            'silhouette': silhouette
        }
        
        # Track best method
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_method = method
    
    print(f"\nBest linkage method: {best_method} (Silhouette: {best_silhouette:.3f})")
    
    return results, best_method

def plot_dendrogram(X_scaled, method='ward'):
    """Plot dendrogram for hierarchical clustering"""
    print(f"\nCreating dendrogram for {method} linkage...")
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X_scaled, method=method)
    
    # Plot dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title(f'Dendrogram - {method.title()} Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'../results/dendrogram_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_clusters(X_scaled, cluster_labels, y_true, method_name, feature_cols):
    """Visualize clustering results using PCA"""
    print(f"\nVisualizing {method_name} clusters...")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. True labels
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('True Labels')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Heart Disease (0: No, 1: Yes)')
    
    # 2. Predicted clusters
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    axes[0, 1].set_title(f'{method_name} Clusters')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    
    # 3. Cluster vs True Labels comparison
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=50)
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.3, s=20)
    axes[1, 0].set_title('Clusters vs True Labels (Overlay)')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # 4. Cluster analysis
    cluster_analysis = pd.DataFrame({
        'cluster': cluster_labels,
        'true_label': y_true
    })
    
    cluster_summary = cluster_analysis.groupby('cluster')['true_label'].agg(['count', 'mean']).round(3)
    cluster_summary.columns = ['Count', 'Heart Disease Rate']
    
    # Create bar plot
    x_pos = np.arange(len(cluster_summary))
    bars = axes[1, 1].bar(x_pos, cluster_summary['Count'], alpha=0.7, label='Count')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Count', color='blue')
    axes[1, 1].tick_params(axis='y', labelcolor='blue')
    
    # Add heart disease rate as line
    ax2 = axes[1, 1].twinx()
    ax2.plot(x_pos, cluster_summary['Heart Disease Rate'], 'ro-', linewidth=2, markersize=8, label='Heart Disease Rate')
    ax2.set_ylabel('Heart Disease Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    axes[1, 1].set_title('Cluster Analysis')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(cluster_summary.index)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, cluster_summary['Count'])):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'../results/{method_name.lower().replace(" ", "_")}_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_summary

def compare_clustering_methods(kmeans_results, hierarchical_results, y_true):
    """Compare different clustering methods"""
    print("\n" + "="*60)
    print("CLUSTERING METHODS COMPARISON")
    print("="*60)
    
    # Prepare comparison data
    methods = ['K-Means']
    silhouettes = [kmeans_results[2]]  # silhouette score
    
    # Add hierarchical methods
    for method, results in hierarchical_results.items():
        methods.append(f'Hierarchical-{method}')
        silhouettes.append(results['silhouette'])
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Method': methods,
        'Silhouette Score': silhouettes
    }).sort_values('Silhouette Score', ascending=False)
    
    print("Clustering Methods Comparison:")
    print(comparison_df)
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(methods, silhouettes)
    plt.title('Silhouette Score Comparison')
    plt.ylabel('Silhouette Score')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, silhouettes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Best method analysis
    best_method = comparison_df.iloc[0]
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'Best Method:\n{best_method["Method"]}\n\nSilhouette Score:\n{best_method["Silhouette Score"]:.3f}',
             ha='center', va='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.title('Best Clustering Method')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def save_clustering_results(kmeans_model, hierarchical_results, best_method, optimal_k):
    """Save clustering models and results"""
    print(f"\nSaving clustering results...")
    
    # Save K-Means model
    joblib.dump(kmeans_model, '../models/kmeans_model.pkl')
    print("K-Means model saved to: models/kmeans_model.pkl")
    
    # Save best hierarchical model
    best_hierarchical = hierarchical_results[best_method]['model']
    joblib.dump(best_hierarchical, '../models/hierarchical_model.pkl')
    print("Best hierarchical model saved to: models/hierarchical_model.pkl")
    
    # Save optimal k
    with open('../results/optimal_k.txt', 'w') as f:
        f.write(str(optimal_k))
    print("Optimal k saved to: results/optimal_k.txt")

def main():
    """Main unsupervised learning pipeline"""
    print("="*60)
    print("UNSUPERVISED LEARNING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X_scaled, y_true, datasets, feature_cols, scaler = prepare_data(df)
    
    # Find optimal number of clusters
    optimal_k, inertias, silhouette_scores = find_optimal_k(X_scaled)
    
    # K-Means clustering
    kmeans_model, kmeans_labels, kmeans_silhouette = kmeans_clustering(X_scaled, optimal_k, y_true)
    
    # Hierarchical clustering
    hierarchical_results, best_hierarchical_method = hierarchical_clustering(X_scaled, optimal_k, y_true)
    
    # Plot dendrogram
    plot_dendrogram(X_scaled, best_hierarchical_method)
    
    # Visualize clusters
    print("\nVisualizing K-Means clusters...")
    kmeans_summary = visualize_clusters(X_scaled, kmeans_labels, y_true, "K-Means", feature_cols)
    
    print(f"\nVisualizing Hierarchical-{best_hierarchical_method} clusters...")
    hierarchical_labels = hierarchical_results[best_hierarchical_method]['labels']
    hierarchical_summary = visualize_clusters(X_scaled, hierarchical_labels, y_true, 
                                            f"Hierarchical-{best_hierarchical_method}", feature_cols)
    
    # Compare methods
    comparison_df = compare_clustering_methods((kmeans_model, kmeans_labels, kmeans_silhouette), 
                                             hierarchical_results, y_true)
    
    # Save results
    save_clustering_results(kmeans_model, hierarchical_results, best_hierarchical_method, optimal_k)
    
    # Save comparison results
    comparison_df.to_csv('../results/clustering_comparison.csv', index=False)
    print("Clustering comparison saved to: results/clustering_comparison.csv")
    
    print("\n" + "="*60)
    print("UNSUPERVISED LEARNING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return kmeans_model, hierarchical_results, comparison_df

if __name__ == "__main__":
    kmeans_model, hierarchical_results, comparison_df = main()
