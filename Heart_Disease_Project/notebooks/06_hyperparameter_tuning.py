"""
Hyperparameter Tuning for Heart Disease Prediction
Using GridSearchCV and RandomizedSearchCV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                   train_test_split, cross_val_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed heart disease data with selected features"""
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
    """Prepare data for hyperparameter tuning"""
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

def define_parameter_grids():
    """Define parameter grids for different models"""
    print("\nDefining parameter grids for hyperparameter tuning...")
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000, 5000]
        },
        
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300, 500],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]
        }
    }
    
    # Create randomized search parameter distributions
    param_distributions = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000, 5000]
        },
        
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 5, 10, 15],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300, 500, 1000],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10, 15],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]
        }
    }
    
    return param_grids, param_distributions

def grid_search_tuning(X_train, y_train, param_grids):
    """Perform GridSearchCV for hyperparameter tuning"""
    print("\n" + "="*60)
    print("GRID SEARCH CV HYPERPARAMETER TUNING")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    grid_results = {}
    
    for name, model in models.items():
        print(f"\nTuning {name} with GridSearchCV...")
        start_time = time.time()
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        # Store results
        grid_results[name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuning_time': tuning_time,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        print(f"  Tuning time: {tuning_time:.2f} seconds")
    
    return grid_results

def randomized_search_tuning(X_train, y_train, param_distributions, n_iter=50):
    """Perform RandomizedSearchCV for hyperparameter tuning"""
    print(f"\n" + "="*60)
    print(f"RANDOMIZED SEARCH CV HYPERPARAMETER TUNING (n_iter={n_iter})")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    random_results = {}
    
    for name, model in models.items():
        print(f"\nTuning {name} with RandomizedSearchCV...")
        start_time = time.time()
        
        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the random search
        random_search.fit(X_train, y_train)
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        # Store results
        random_results[name] = {
            'best_estimator': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'tuning_time': tuning_time,
            'cv_results': random_search.cv_results_
        }
        
        print(f"  Best parameters: {random_search.best_params_}")
        print(f"  Best CV score: {random_search.best_score_:.4f}")
        print(f"  Tuning time: {tuning_time:.2f} seconds")
    
    return random_results

def evaluate_tuned_models(grid_results, random_results, X_test, y_test):
    """Evaluate tuned models on test set"""
    print("\n" + "="*60)
    print("EVALUATING TUNED MODELS")
    print("="*60)
    
    evaluation_results = {}
    
    # Evaluate GridSearch results
    print("\nGridSearchCV Results:")
    for name, results in grid_results.items():
        model = results['best_estimator']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        test_accuracy = model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        evaluation_results[f'{name}_Grid'] = {
            'model': model,
            'best_params': results['best_params'],
            'cv_score': results['best_score'],
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'tuning_time': results['tuning_time']
        }
        
        print(f"  {name}:")
        print(f"    CV Score: {results['best_score']:.4f}")
        print(f"    Test Accuracy: {test_accuracy:.4f}")
        print(f"    Test AUC: {test_auc:.4f}" if test_auc else "    Test AUC: N/A")
        print(f"    Tuning Time: {results['tuning_time']:.2f}s")
    
    # Evaluate RandomizedSearch results
    print("\nRandomizedSearchCV Results:")
    for name, results in random_results.items():
        model = results['best_estimator']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        test_accuracy = model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        evaluation_results[f'{name}_Random'] = {
            'model': model,
            'best_params': results['best_params'],
            'cv_score': results['best_score'],
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'tuning_time': results['tuning_time']
        }
        
        print(f"  {name}:")
        print(f"    CV Score: {results['best_score']:.4f}")
        print(f"    Test Accuracy: {test_accuracy:.4f}")
        print(f"    Test AUC: {test_auc:.4f}" if test_auc else "    Test AUC: N/A")
        print(f"    Tuning Time: {results['tuning_time']:.2f}s")
    
    return evaluation_results

def compare_tuning_methods(evaluation_results):
    """Compare GridSearchCV vs RandomizedSearchCV"""
    print("\n" + "="*60)
    print("COMPARING TUNING METHODS")
    print("="*60)
    
    # Create comparison dataframe
    comparison_data = []
    
    for name, results in evaluation_results.items():
        model_name = name.split('_')[0]
        tuning_method = name.split('_')[1]
        
        comparison_data.append({
            'Model': model_name,
            'Tuning Method': tuning_method,
            'CV Score': results['cv_score'],
            'Test Accuracy': results['test_accuracy'],
            'Test AUC': results['test_auc'] if results['test_auc'] else 0,
            'Tuning Time (s)': results['tuning_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Tuning Methods Comparison:")
    print(comparison_df.round(4))
    
    # Find best model overall
    best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
    print(f"\nBest Model Overall:")
    print(f"  Model: {best_model['Model']}")
    print(f"  Tuning Method: {best_model['Tuning Method']}")
    print(f"  Test Accuracy: {best_model['Test Accuracy']:.4f}")
    print(f"  Test AUC: {best_model['Test AUC']:.4f}")
    print(f"  Tuning Time: {best_model['Tuning Time (s)']:.2f}s")
    
    return comparison_df, best_model

def visualize_tuning_results(evaluation_results, comparison_df):
    """Create visualizations for tuning results"""
    print("\nCreating tuning results visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Test Accuracy Comparison
    ax1 = axes[0, 0]
    model_names = comparison_df['Model'].unique()
    grid_scores = []
    random_scores = []
    
    for model in model_names:
        grid_score = comparison_df[(comparison_df['Model'] == model) & 
                                 (comparison_df['Tuning Method'] == 'Grid')]['Test Accuracy'].iloc[0]
        random_score = comparison_df[(comparison_df['Model'] == model) & 
                                   (comparison_df['Tuning Method'] == 'Random')]['Test Accuracy'].iloc[0]
        grid_scores.append(grid_score)
        random_scores.append(random_score)
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, grid_scores, width, label='GridSearchCV', alpha=0.8)
    ax1.bar(x + width/2, random_scores, width, label='RandomizedSearchCV', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Tuning Time Comparison
    ax2 = axes[0, 1]
    grid_times = []
    random_times = []
    
    for model in model_names:
        grid_time = comparison_df[(comparison_df['Model'] == model) & 
                                (comparison_df['Tuning Method'] == 'Grid')]['Tuning Time (s)'].iloc[0]
        random_time = comparison_df[(comparison_df['Model'] == model) & 
                                  (comparison_df['Tuning Method'] == 'Random')]['Tuning Time (s)'].iloc[0]
        grid_times.append(grid_time)
        random_times.append(random_time)
    
    ax2.bar(x - width/2, grid_times, width, label='GridSearchCV', alpha=0.8)
    ax2.bar(x + width/2, random_times, width, label='RandomizedSearchCV', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Tuning Time (seconds)')
    ax2.set_title('Tuning Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. CV Score vs Test Accuracy
    ax3 = axes[0, 2]
    scatter = ax3.scatter(comparison_df['CV Score'], comparison_df['Test Accuracy'], 
                         c=comparison_df['Tuning Time (s)'], cmap='viridis', s=100, alpha=0.7)
    ax3.set_xlabel('CV Score')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('CV Score vs Test Accuracy')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Tuning Time (s)')
    
    # 4. Model Performance Heatmap
    ax4 = axes[1, 0]
    pivot_table = comparison_df.pivot(index='Model', columns='Tuning Method', values='Test Accuracy')
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', ax=ax4, fmt='.4f')
    ax4.set_title('Test Accuracy Heatmap')
    
    # 5. Time Efficiency (Accuracy per second)
    ax5 = axes[1, 1]
    comparison_df['Efficiency'] = comparison_df['Test Accuracy'] / comparison_df['Tuning Time (s)']
    efficiency_pivot = comparison_df.pivot(index='Model', columns='Tuning Method', values='Efficiency')
    sns.heatmap(efficiency_pivot, annot=True, cmap='Blues', ax=ax5, fmt='.4f')
    ax5.set_title('Time Efficiency (Accuracy/Time)')
    
    # 6. Best Parameters Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Get best model details
    best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
    best_method = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Tuning Method']
    best_key = f"{best_model_name}_{best_method}"
    best_params = evaluation_results[best_key]['best_params']
    
    params_text = f"Best Model: {best_model_name}\n"
    params_text += f"Tuning Method: {best_method}\n"
    params_text += f"Test Accuracy: {comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Test Accuracy']:.4f}\n\n"
    params_text += "Best Parameters:\n"
    for param, value in best_params.items():
        params_text += f"  {param}: {value}\n"
    
    ax6.text(0.1, 0.9, params_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax6.set_title('Best Model Summary')
    
    plt.tight_layout()
    plt.savefig('../results/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_tuned_models(evaluation_results, best_model_info):
    """Save the best tuned models"""
    print(f"\nSaving tuned models...")
    
    # Save all tuned models
    for name, results in evaluation_results.items():
        model_name = name.lower().replace(' ', '_')
        joblib.dump(results['model'], f'../models/tuned_{model_name}.pkl')
        print(f"  Tuned {name} model saved to: models/tuned_{model_name}.pkl")
    
    # Save the best model separately
    best_model_name = best_model_info['Model']
    best_method = best_model_info['Tuning Method']
    best_key = f"{best_model_name}_{best_method}"
    best_model = evaluation_results[best_key]['model']
    
    joblib.dump(best_model, '../models/best_tuned_model.pkl')
    print(f"  Best tuned model saved to: models/best_tuned_model.pkl")
    
    # Save best parameters
    best_params = evaluation_results[best_key]['best_params']
    with open('../results/best_parameters.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Tuning Method: {best_method}\n")
        f.write(f"Test Accuracy: {best_model_info['Test Accuracy']:.4f}\n")
        f.write(f"Test AUC: {best_model_info['Test AUC']:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    print("  Best parameters saved to: results/best_parameters.txt")

def main():
    """Main hyperparameter tuning pipeline"""
    print("="*60)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # Define parameter grids
    param_grids, param_distributions = define_parameter_grids()
    
    # GridSearchCV tuning
    print("\nStarting GridSearchCV tuning...")
    grid_results = grid_search_tuning(X_train, y_train, param_grids)
    
    # RandomizedSearchCV tuning
    print("\nStarting RandomizedSearchCV tuning...")
    random_results = randomized_search_tuning(X_train, y_train, param_distributions, n_iter=50)
    
    # Evaluate tuned models
    evaluation_results = evaluate_tuned_models(grid_results, random_results, X_test, y_test)
    
    # Compare tuning methods
    comparison_df, best_model_info = compare_tuning_methods(evaluation_results)
    
    # Create visualizations
    visualize_tuning_results(evaluation_results, comparison_df)
    
    # Save results
    save_tuned_models(evaluation_results, best_model_info)
    
    # Save comparison results
    comparison_df.to_csv('../results/hyperparameter_tuning_comparison.csv', index=False)
    print("Hyperparameter tuning comparison saved to: results/hyperparameter_tuning_comparison.csv")
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return evaluation_results, comparison_df, best_model_info

if __name__ == "__main__":
    evaluation_results, comparison_df, best_model_info = main()
