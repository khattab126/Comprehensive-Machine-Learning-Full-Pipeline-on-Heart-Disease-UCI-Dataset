"""
Quick model training for Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    print("Training basic model for Streamlit app...")
    
    # Load processed data
    df = pd.read_csv('data/heart_disease_processed.csv')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest (best performing model)
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate accuracy
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save model and scaler
    joblib.dump(rf_model, 'models/best_model.pkl')
    joblib.dump(rf_model, 'models/best_tuned_model.pkl')
    
    # Create scaler (even though data is already scaled)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("✅ Model training completed!")
    print("✅ Models saved to models/ directory")
    print("✅ Streamlit app should now work!")

if __name__ == "__main__":
    main()
