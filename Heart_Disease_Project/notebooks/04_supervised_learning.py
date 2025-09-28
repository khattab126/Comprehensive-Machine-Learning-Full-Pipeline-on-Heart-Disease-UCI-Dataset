"""
Supervised Learning Models for Heart Disease Prediction
Logistic Regression, Decision Tree, Random Forest, SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, 
                           accuracy_score, precision_score, recall_score, f1_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed heart disease data with selected features"""
    try:
        df = pd.read_csv('../data/heart_disease_selected_features.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Selected features data not found. Using full processed data...")
        try:
            df = pd.read_csv('../data/heart_disease_processed.csv')
            # Use all features except target and dataset
            feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
            df = df[feature_cols + ['target', 'dataset']]
            print(f"Full data loaded. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Processed data not found. Please run data preprocessing first.")
            return None

def prepare_data(df):
    """Prepare data for model training"""
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Features: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_models(X_train, X_test, y_train, y_test):
    """Train all supervised learning models"""
    print("="*60)
    print("TRAINING SUPERVISED LEARNING MODELS")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store results
        model_results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }
        
        trained_models[name] = model
        
        # Print results
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model_results, trained_models

def evaluate_models(model_results, y_test):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        name: {
            'Test Accuracy': results['test_accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'CV Mean': results['cv_mean'],
            'CV Std': results['cv_std']
        }
        for name, results in model_results.items()
    }).T
    
    print("Model Performance Summary:")
    print(results_df.round(4))
    
    # Find best model
    best_model_name = results_df['Test Accuracy'].idxmax()
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best test accuracy: {results_df.loc[best_model_name, 'Test Accuracy']:.4f}")
    
    return results_df, best_model_name

def plot_model_comparison(model_results, y_test):
    """Create comprehensive model comparison plots"""
    print("\nCreating model comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract metrics for plotting
    model_names = list(model_results.keys())
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
    metric_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean']
    
    # Bar plots for each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i//3, i%3]
        values = [model_results[name][metric] for name in model_names]
        bars = ax.bar(model_names, values)
        ax.set_title(f'{label} Comparison')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # 6. Confusion Matrices
    ax = axes[1, 2]
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_accuracy'])
    cm = confusion_matrix(y_test, model_results[best_model_name]['y_test_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC Curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for name, results in model_results.items():
        if results['y_test_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results['y_test_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # 3. Precision-Recall Curves
    plt.subplot(2, 2, 2)
    for name, results in model_results.items():
        if results['y_test_proba'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, results['y_test_proba'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    
    # 4. Cross-Validation Scores
    plt.subplot(2, 2, 3)
    cv_means = [model_results[name]['cv_mean'] for name in model_names]
    cv_stds = [model_results[name]['cv_std'] for name in model_names]
    
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
    plt.title('Cross-Validation Scores')
    plt.ylabel('CV Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # 5. Feature Importance (for tree-based models)
    plt.subplot(2, 2, 4)
    # This will be populated if we have feature importance
    plt.text(0.5, 0.5, 'Feature Importance\n(Will be shown in detailed analysis)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_model_analysis(trained_models, X_train, y_train, feature_cols):
    """Detailed analysis of individual models"""
    print("\n" + "="*60)
    print("DETAILED MODEL ANALYSIS")
    print("="*60)
    
    # 1. Feature Importance Analysis
    plt.figure(figsize=(15, 10))
    
    # Random Forest Feature Importance
    if 'Random Forest' in trained_models:
        plt.subplot(2, 2, 1)
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        feature_importance.plot(x='feature', y='importance', kind='barh', ax=plt.gca())
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance Score')
    
    # Decision Tree Feature Importance
    if 'Decision Tree' in trained_models:
        plt.subplot(2, 2, 2)
        dt_model = trained_models['Decision Tree']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        feature_importance.plot(x='feature', y='importance', kind='barh', ax=plt.gca())
        plt.title('Decision Tree Feature Importance')
        plt.xlabel('Importance Score')
    
    # Logistic Regression Coefficients
    if 'Logistic Regression' in trained_models:
        plt.subplot(2, 2, 3)
        lr_model = trained_models['Logistic Regression']
        coefficients = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': lr_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=True)
        
        coefficients.plot(x='feature', y='coefficient', kind='barh', ax=plt.gca())
        plt.title('Logistic Regression Coefficients')
        plt.xlabel('Coefficient Value')
    
    # Model Complexity Analysis
    plt.subplot(2, 2, 4)
    model_complexity = {
        'Logistic Regression': 1,  # Linear
        'Decision Tree': 2,  # Medium
        'Random Forest': 3,  # High
        'SVM': 2  # Medium-High
    }
    
    complexity_df = pd.DataFrame(list(model_complexity.items()), columns=['Model', 'Complexity'])
    complexity_df.plot(x='Model', y='Complexity', kind='bar', ax=plt.gca())
    plt.title('Model Complexity Comparison')
    plt.ylabel('Complexity Level')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/detailed_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_models(trained_models, best_model_name):
    """Save trained models"""
    print(f"\nSaving trained models...")
    
    # Save all models
    for name, model in trained_models.items():
        joblib.dump(model, f'../models/{name.lower().replace(" ", "_")}_model.pkl')
        print(f"  {name} model saved to: models/{name.lower().replace(' ', '_')}_model.pkl")
    
    # Save best model separately
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, '../models/best_model.pkl')
    print(f"  Best model ({best_model_name}) saved to: models/best_model.pkl")

def main():
    """Main supervised learning pipeline"""
    print("="*60)
    print("SUPERVISED LEARNING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # Train models
    model_results, trained_models = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    results_df, best_model_name = evaluate_models(model_results, y_test)
    
    # Create visualizations
    plot_model_comparison(model_results, y_test)
    
    # Detailed analysis
    detailed_model_analysis(trained_models, X_train, y_train, feature_cols)
    
    # Save models
    save_models(trained_models, best_model_name)
    
    # Save results
    results_df.to_csv('../results/supervised_learning_results.csv')
    print(f"\nResults saved to: results/supervised_learning_results.csv")
    
    print("\n" + "="*60)
    print("SUPERVISED LEARNING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model_results, trained_models, results_df, best_model_name

if __name__ == "__main__":
    model_results, trained_models, results_df, best_model_name = main()
