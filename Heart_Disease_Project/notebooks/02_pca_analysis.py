"""
PCA Analysis for Heart Disease Dataset
Dimensionality Reduction and Variance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the preprocessed heart disease data"""
    try:
        df = pd.read_csv('../data/heart_disease_processed.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Processed data not found. Please run data preprocessing first.")
        return None

def perform_pca_analysis(df):
    """Perform comprehensive PCA analysis"""
    print("="*60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*60)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Original feature space: {X.shape[1]} features")
    print(f"Number of samples: {X.shape[0]}")
    
    # 1. Determine optimal number of components
    print("\n1. Determining optimal number of components...")
    
    # Fit PCA with all components to analyze explained variance
    pca_full = PCA()
    pca_full.fit(X)
    
    # Calculate cumulative explained variance
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
    n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
    n_components_85 = np.argmax(cumsum_variance >= 0.85) + 1
    
    print(f"Components for 85% variance: {n_components_85}")
    print(f"Components for 90% variance: {n_components_90}")
    print(f"Components for 95% variance: {n_components_95}")
    
    # 2. Visualize explained variance
    print("\n2. Creating variance analysis plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Individual explained variance
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             pca_full.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True)
    
    # Cumulative explained variance
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'ro-')
    plt.axhline(y=0.85, color='g', linestyle='--', label='85% Variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    # Scree plot
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             pca_full.explained_variance_ratio_, 'go-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True)
    
    # Feature importance in first two components
    plt.subplot(2, 2, 4)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    # Plot first two principal components
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('First Two Principal Components')
    plt.colorbar(scatter, label='Heart Disease (0: No, 1: Yes)')
    
    plt.tight_layout()
    plt.savefig('../results/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Apply PCA with optimal number of components
    print(f"\n3. Applying PCA with {n_components_90} components (90% variance)...")
    
    pca_optimal = PCA(n_components=n_components_90)
    X_pca = pca_optimal.fit_transform(X)
    
    print(f"Reduced feature space: {X_pca.shape[1]} components")
    print(f"Variance retained: {pca_optimal.explained_variance_ratio_.sum():.2%}")
    
    # 4. Create PCA dataframe
    pca_columns = [f'PC{i+1}' for i in range(n_components_90)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['target'] = y.values
    df_pca['dataset'] = df['dataset'].values
    
    # 5. Analyze component loadings
    print("\n4. Analyzing component loadings...")
    
    # Create loadings dataframe
    loadings_df = pd.DataFrame(
        pca_optimal.components_.T,
        columns=pca_columns,
        index=feature_cols
    )
    
    print("Component loadings (first 5 components):")
    print(loadings_df.iloc[:, :5].round(3))
    
    # Visualize loadings
    plt.figure(figsize=(12, 8))
    
    # Heatmap of loadings for first 5 components
    plt.subplot(2, 1, 1)
    sns.heatmap(loadings_df.iloc[:, :5], annot=True, cmap='RdBu_r', center=0)
    plt.title('Component Loadings (First 5 Components)')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    
    # Bar plot of loadings for PC1
    plt.subplot(2, 1, 2)
    loadings_df['PC1'].abs().sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Importance in PC1 (Absolute Loadings)')
    plt.xlabel('Absolute Loading Value')
    
    plt.tight_layout()
    plt.savefig('../results/pca_loadings.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Save PCA results
    print("\n5. Saving PCA results...")
    
    # Save PCA-transformed data
    df_pca.to_csv('../data/heart_disease_pca.csv', index=False)
    print(f"PCA-transformed data saved to: data/heart_disease_pca.csv")
    
    # Save PCA model
    joblib.dump(pca_optimal, '../models/pca_model.pkl')
    print(f"PCA model saved to: models/pca_model.pkl")
    
    # Save loadings
    loadings_df.to_csv('../results/pca_loadings.csv')
    print(f"Component loadings saved to: results/pca_loadings.csv")
    
    # 7. Summary statistics
    print("\n" + "="*40)
    print("PCA ANALYSIS SUMMARY")
    print("="*40)
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Variance retained: {pca_optimal.explained_variance_ratio_.sum():.2%}")
    print(f"Reduction ratio: {X_pca.shape[1]/X.shape[1]:.2%}")
    
    print("\nExplained variance by component:")
    for i, var in enumerate(pca_optimal.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    return df_pca, pca_optimal, loadings_df

def main():
    """Main PCA analysis pipeline"""
    # Load processed data
    df = load_processed_data()
    if df is None:
        return
    
    # Perform PCA analysis
    df_pca, pca_model, loadings = perform_pca_analysis(df)
    
    print("\n" + "="*60)
    print("PCA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df_pca, pca_model, loadings

if __name__ == "__main__":
    df_pca, pca_model, loadings = main()
