"""
Heart Disease Data Preprocessing Script
Combines all heart disease datasets and performs comprehensive preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Define column names for the 14 attributes used in heart disease prediction
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def load_and_combine_datasets():
    """Load and combine all heart disease datasets"""
    print("Loading heart disease datasets...")
    
    # Load all processed datasets
    datasets = []
    
    # Cleveland dataset
    try:
        cleveland = pd.read_csv('../processed.cleveland.data', names=column_names, na_values='?')
        cleveland['dataset'] = 'cleveland'
        datasets.append(cleveland)
        print(f"Cleveland dataset loaded: {cleveland.shape}")
    except Exception as e:
        print(f"Error loading Cleveland dataset: {e}")
    
    # Hungarian dataset
    try:
        hungarian = pd.read_csv('../processed.hungarian.data', names=column_names, na_values='?')
        hungarian['dataset'] = 'hungarian'
        datasets.append(hungarian)
        print(f"Hungarian dataset loaded: {hungarian.shape}")
    except Exception as e:
        print(f"Error loading Hungarian dataset: {e}")
    
    # Switzerland dataset
    try:
        switzerland = pd.read_csv('../processed.switzerland.data', names=column_names, na_values='?')
        switzerland['dataset'] = 'switzerland'
        datasets.append(switzerland)
        print(f"Switzerland dataset loaded: {switzerland.shape}")
    except Exception as e:
        print(f"Error loading Switzerland dataset: {e}")
    
    # VA dataset
    try:
        va = pd.read_csv('../processed.va.data', names=column_names, na_values='?')
        va['dataset'] = 'va'
        datasets.append(va)
        print(f"VA dataset loaded: {va.shape}")
    except Exception as e:
        print(f"Error loading VA dataset: {e}")
    
    # Combine all datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        return combined_df
    else:
        raise Exception("No datasets could be loaded")

def preprocess_data(df):
    """Comprehensive data preprocessing"""
    print("\nStarting data preprocessing...")
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # 1. Handle missing values
    print("1. Handling missing values...")
    print(f"Missing values before preprocessing:\n{df_processed.isnull().sum()}")
    
    # Replace missing values with median for numerical columns
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Replace missing values with mode for categorical columns
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    print(f"Missing values after preprocessing:\n{df_processed.isnull().sum()}")
    
    # 2. Convert target variable to binary (0: no disease, 1: disease)
    print("\n2. Converting target variable to binary...")
    df_processed['target'] = df_processed['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f"Target distribution:\n{df_processed['target'].value_counts()}")
    
    # 3. Data encoding (already numerical, but ensure proper types)
    print("\n3. Ensuring proper data types...")
    for col in df_processed.columns:
        if col != 'dataset':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # 4. Remove any remaining NaN values
    df_processed = df_processed.dropna()
    
    # 5. Feature scaling
    print("\n4. Applying feature scaling...")
    scaler = StandardScaler()
    feature_cols = [col for col in df_processed.columns if col not in ['target', 'dataset']]
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    
    print(f"Final dataset shape: {df_processed.shape}")
    return df_processed, scaler

def exploratory_data_analysis(df):
    """Perform comprehensive EDA"""
    print("\nPerforming Exploratory Data Analysis...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Dataset overview
    print(f"\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # 2. Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # 3. Target distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Heart Disease (0: No, 1: Yes)')
    plt.ylabel('Count')
    
    # 4. Age distribution
    plt.subplot(2, 3, 2)
    df['age'].hist(bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # 5. Sex distribution
    plt.subplot(2, 3, 3)
    df['sex'].value_counts().plot(kind='bar')
    plt.title('Sex Distribution')
    plt.xlabel('Sex (0: Female, 1: Male)')
    plt.ylabel('Count')
    
    # 6. Chest pain type distribution
    plt.subplot(2, 3, 4)
    df['cp'].value_counts().plot(kind='bar')
    plt.title('Chest Pain Type Distribution')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Count')
    
    # 7. Correlation heatmap
    plt.subplot(2, 3, 5)
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    correlation_matrix = df[feature_cols + ['target']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    # 8. Box plot for numerical features
    plt.subplot(2, 3, 6)
    df[['age', 'trestbps', 'chol', 'thalach']].boxplot()
    plt.title('Numerical Features Box Plot')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Dataset distribution
    plt.figure(figsize=(10, 6))
    df['dataset'].value_counts().plot(kind='bar')
    plt.title('Dataset Distribution')
    plt.xlabel('Dataset')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("HEART DISEASE DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load and combine datasets
    df = load_and_combine_datasets()
    
    # Perform EDA on raw data
    print("\n" + "="*40)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*40)
    df_eda = exploratory_data_analysis(df)
    
    # Preprocess the data
    print("\n" + "="*40)
    print("DATA PREPROCESSING")
    print("="*40)
    df_processed, scaler = preprocess_data(df)
    
    # Save processed data
    df_processed.to_csv('data/heart_disease_processed.csv', index=False)
    print(f"\nProcessed data saved to: data/heart_disease_processed.csv")
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Scaler saved to: models/scaler.pkl")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df_processed, scaler

if __name__ == "__main__":
    df_processed, scaler = main()
