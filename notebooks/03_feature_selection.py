"""
Feature Selection for Heart Disease Dataset
Using RFE, Chi-Square Test, and Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed heart disease data"""
    try:
        df = pd.read_csv('../data/heart_disease_processed.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Processed data not found. Please run data preprocessing first.")
        return None

def prepare_data(df):
    """Prepare data for feature selection"""
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Features for selection: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def random_forest_feature_importance(X, y, feature_cols):
    """Calculate feature importance using Random Forest"""
    print("\n" + "="*50)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*50)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_scores = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance Scores:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=plt.gca())
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Select top features (e.g., top 8)
    top_features_rf = feature_importance.head(8)['feature'].tolist()
    print(f"\nTop 8 features by Random Forest: {top_features_rf}")
    
    return feature_importance, top_features_rf

def recursive_feature_elimination(X, y, feature_cols):
    """Perform Recursive Feature Elimination"""
    print("\n" + "="*50)
    print("RECURSIVE FEATURE ELIMINATION (RFE)")
    print("="*50)
    
    # Use Logistic Regression as base estimator
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # RFE with different numbers of features
    n_features_list = [5, 6, 7, 8, 9, 10]
    rfe_results = []
    
    for n_features in n_features_list:
        rfe = RFE(estimator=lr, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        # Get selected features
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
        
        # Evaluate performance
        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        lr_selected = LogisticRegression(random_state=42, max_iter=1000)
        lr_selected.fit(X_train, y_train)
        score = lr_selected.score(X_test, y_test)
        
        rfe_results.append({
            'n_features': n_features,
            'features': selected_features,
            'score': score
        })
        
        print(f"RFE with {n_features} features: {score:.4f}")
        print(f"Selected features: {selected_features}")
        print()
    
    # Find best RFE result
    best_rfe = max(rfe_results, key=lambda x: x['score'])
    print(f"Best RFE result: {best_rfe['n_features']} features with score {best_rfe['score']:.4f}")
    print(f"Best RFE features: {best_rfe['features']}")
    
    return rfe_results, best_rfe

def chi_square_test(X, y, feature_cols):
    """Perform Chi-Square test for feature selection"""
    print("\n" + "="*50)
    print("CHI-SQUARE TEST FOR FEATURE SELECTION")
    print("="*50)
    
    # Chi-square test requires non-negative values, so scale to [0, 1]
    scaler = MinMaxScaler()
    X_nonneg = scaler.fit_transform(X)
    chi2_scores, p_values = chi2(X_nonneg, y)
    
    # Create results dataframe
    chi2_results = pd.DataFrame({
        'feature': feature_cols,
        'chi2_score': chi2_scores,
        'p_value': p_values
    }).sort_values('chi2_score', ascending=False)
    
    print("Chi-Square Test Results:")
    print(chi2_results)
    
    # Select features with p-value < 0.05
    significant_features = chi2_results[chi2_results['p_value'] < 0.05]['feature'].tolist()
    print(f"\nSignificant features (p < 0.05): {significant_features}")
    
    # Visualize chi-square scores
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    chi2_results.plot(x='feature', y='chi2_score', kind='barh', ax=plt.gca())
    plt.title('Chi-Square Scores')
    plt.xlabel('Chi-Square Score')
    
    plt.subplot(1, 2, 2)
    chi2_results.plot(x='feature', y='p_value', kind='barh', ax=plt.gca())
    plt.axvline(x=0.05, color='r', linestyle='--', label='p = 0.05')
    plt.title('P-Values')
    plt.xlabel('P-Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/chi_square_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return chi2_results, significant_features

def select_k_best_features(X, y, feature_cols, k=8):
    """Select K best features using SelectKBest"""
    print(f"\n" + "="*50)
    print(f"SELECT K BEST FEATURES (K={k})")
    print("="*50)
    
    # Select K best features using chi-square (requires non-negative values)
    scaler = MinMaxScaler()
    X_nonneg = scaler.fit_transform(X)
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X_nonneg, y)
    
    # Get selected feature names
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    print(f"Selected {k} best features: {selected_features}")
    
    return selected_features, selector

def comprehensive_feature_selection(X, y, feature_cols):
    """Perform comprehensive feature selection analysis"""
    print("="*60)
    print("COMPREHENSIVE FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # 1. Random Forest Feature Importance
    rf_importance, top_rf_features = random_forest_feature_importance(X, y, feature_cols)
    
    # 2. Recursive Feature Elimination
    rfe_results, best_rfe = recursive_feature_elimination(X, y, feature_cols)
    
    # 3. Chi-Square Test
    chi2_results, significant_chi2 = chi_square_test(X, y, feature_cols)
    
    # 4. Select K Best
    k_best_features, selector = select_k_best_features(X, y, feature_cols, k=8)
    
    # 5. Combine results
    print("\n" + "="*50)
    print("FEATURE SELECTION SUMMARY")
    print("="*50)
    
    # Create feature selection summary
    feature_summary = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': [rf_importance[rf_importance['feature'] == f]['importance'].iloc[0] for f in feature_cols],
        'chi2_score': [chi2_results[chi2_results['feature'] == f]['chi2_score'].iloc[0] for f in feature_cols],
        'chi2_p_value': [chi2_results[chi2_results['feature'] == f]['p_value'].iloc[0] for f in feature_cols],
        'rfe_selected': [f in best_rfe['features'] for f in feature_cols],
        'k_best_selected': [f in k_best_features for f in feature_cols]
    })
    
    # Calculate selection score (how many methods selected each feature)
    feature_summary['selection_score'] = (
        feature_summary['rfe_selected'].astype(int) + 
        feature_summary['k_best_selected'].astype(int) +
        (feature_summary['chi2_p_value'] < 0.05).astype(int)
    )
    
    # Sort by selection score and RF importance
    feature_summary = feature_summary.sort_values(['selection_score', 'rf_importance'], ascending=[False, False])
    
    print("Feature Selection Summary:")
    print(feature_summary)
    
    # Select final features (top 8 based on selection score and importance)
    final_features = feature_summary.head(8)['feature'].tolist()
    print(f"\nFinal selected features (8): {final_features}")
    
    # Visualize feature selection results
    plt.figure(figsize=(15, 10))
    
    # RF Importance
    plt.subplot(2, 2, 1)
    rf_importance.head(8).plot(x='feature', y='importance', kind='barh', ax=plt.gca())
    plt.title('Top 8 Features by RF Importance')
    plt.xlabel('Importance Score')
    
    # Chi-Square Scores
    plt.subplot(2, 2, 2)
    chi2_results.head(8).plot(x='feature', y='chi2_score', kind='barh', ax=plt.gca())
    plt.title('Top 8 Features by Chi-Square Score')
    plt.xlabel('Chi-Square Score')
    
    # Selection Score
    plt.subplot(2, 2, 3)
    feature_summary.head(8).plot(x='feature', y='selection_score', kind='barh', ax=plt.gca())
    plt.title('Feature Selection Score')
    plt.xlabel('Selection Score')
    
    # P-Values
    plt.subplot(2, 2, 4)
    chi2_results.head(8).plot(x='feature', y='p_value', kind='barh', ax=plt.gca())
    plt.axvline(x=0.05, color='r', linestyle='--', label='p = 0.05')
    plt.title('P-Values (Significance)')
    plt.xlabel('P-Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/feature_selection_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_summary, final_features

def evaluate_selected_features(X, y, selected_features):
    """Evaluate performance with selected features"""
    print("\n" + "="*50)
    print("EVALUATING SELECTED FEATURES")
    print("="*50)
    
    # Prepare data with selected features
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        results[name] = {
            'train_score': train_score,
            'test_score': test_score
        }
        
        print(f"{name}:")
        print(f"  Train Score: {train_score:.4f}")
        print(f"  Test Score: {test_score:.4f}")
        print()
    
    return results

def main():
    """Main feature selection pipeline"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X, y, feature_cols = prepare_data(df)
    
    # Perform comprehensive feature selection
    feature_summary, final_features = comprehensive_feature_selection(X, y, feature_cols)
    
    # Evaluate selected features
    evaluation_results = evaluate_selected_features(X, y, final_features)
    
    # Save results
    print("\nSaving feature selection results...")
    
    # Save feature summary
    feature_summary.to_csv('../results/feature_selection_summary.csv', index=False)
    print("Feature selection summary saved to: results/feature_selection_summary.csv")
    
    # Save selected features
    selected_features_df = pd.DataFrame({'selected_features': final_features})
    selected_features_df.to_csv('../results/selected_features.csv', index=False)
    print("Selected features saved to: results/selected_features.csv")
    
    # Create final dataset with selected features
    df_selected = df[final_features + ['target', 'dataset']].copy()
    df_selected.to_csv('../data/heart_disease_selected_features.csv', index=False)
    print("Final dataset with selected features saved to: data/heart_disease_selected_features.csv")
    
    print("\n" + "="*60)
    print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return feature_summary, final_features, df_selected

if __name__ == "__main__":
    feature_summary, final_features, df_selected = main()
