# Heart Disease Prediction System

A comprehensive machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for predicting heart disease risk based on various health parameters. The system includes data preprocessing, feature selection, model training, hyperparameter optimization, and a user-friendly web interface.

## 📊 Dataset

The project uses the UCI Heart Disease dataset, which includes data from four different sources:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- V.A. Medical Center, Long Beach, CA
- University Hospital, Zurich, Switzerland

**Features Used (14 attributes):**
1. Age
2. Sex
3. Chest Pain Type
4. Resting Blood Pressure
5. Serum Cholesterol
6. Fasting Blood Sugar
7. Resting ECG
8. Maximum Heart Rate
9. Exercise Induced Angina
10. ST Depression
11. Slope of Peak Exercise ST Segment
12. Number of Major Vessels
13. Thalassemia
14. Target (Heart Disease)

## 🚀 Features

### Data Processing
- ✅ Data preprocessing and cleaning
- ✅ Missing value imputation
- ✅ Feature scaling and encoding
- ✅ Exploratory data analysis (EDA)

### Dimensionality Reduction
- ✅ Principal Component Analysis (PCA)
- ✅ Variance analysis and visualization
- ✅ Optimal component selection

### Feature Selection
- ✅ Random Forest feature importance
- ✅ Recursive Feature Elimination (RFE)
- ✅ Chi-Square test for feature significance
- ✅ Comprehensive feature ranking

### Supervised Learning
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)
- ✅ Model evaluation and comparison

### Unsupervised Learning
- ✅ K-Means Clustering
- ✅ Hierarchical Clustering
- ✅ Cluster analysis and visualization

### Model Optimization
- ✅ GridSearchCV hyperparameter tuning
- ✅ RandomizedSearchCV optimization
- ✅ Performance comparison and analysis

### Deployment
- ✅ Streamlit web application
- ✅ Real-time prediction interface
- ✅ Interactive data visualization
- ✅ Model export and persistence

## 📁 Project Structure

```
Heart_Disease_Project/
├── data/
│   ├── heart_disease_processed.csv
│   ├── heart_disease_pca.csv
│   └── heart_disease_selected_features.csv
├── notebooks/
│   ├── 02_pca_analysis.py
│   ├── 03_feature_selection.py
│   ├── 04_supervised_learning.py
│   ├── 05_unsupervised_learning.py
│   └── 06_hyperparameter_tuning.py
├── models/
│   ├── best_model.pkl
│   ├── best_tuned_model.pkl
│   ├── scaler.pkl
│   └── pca_model.pkl
├── ui/
│   └── app.py
├── results/
│   ├── eda_analysis.png
│   ├── pca_analysis.png
│   ├── feature_selection_summary.png
│   ├── model_comparison.png
│   └── hyperparameter_tuning_results.png
├── data_preprocessing.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Heart_Disease_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline:**
   ```bash
   # Data preprocessing
   python data_preprocessing.py
   
   # PCA analysis
   python notebooks/02_pca_analysis.py
   
   # Feature selection
   python notebooks/03_feature_selection.py
   
   # Supervised learning
   python notebooks/04_supervised_learning.py
   
   # Unsupervised learning
   python notebooks/05_unsupervised_learning.py
   
   # Hyperparameter tuning
   python notebooks/06_hyperparameter_tuning.py
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run ui/app.py
   ```

## 🎮 Usage

### Web Application
1. Open the Streamlit application in your browser
2. Navigate to the "Prediction" page
3. Enter patient information using the interactive controls
4. Click "Predict" to get the heart disease risk assessment
5. Explore the "Data Analysis" page for insights and visualizations

### Programmatic Usage
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/best_tuned_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input data
input_data = {
    'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

# Make prediction
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
print(f"Probability: {probability}")
```

## 📈 Model Performance

The system achieves high accuracy through comprehensive hyperparameter tuning:

- **Best Model**: Optimized through GridSearchCV and RandomizedSearchCV
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Feature Selection**: Top 8 most important features selected

## 🔬 Technical Details

### Preprocessing Pipeline
1. Data loading and combination from multiple sources
2. Missing value imputation using median/mode
3. Target variable conversion to binary classification
4. Feature scaling using StandardScaler
5. Comprehensive exploratory data analysis

### Feature Selection Methods
1. **Random Forest Importance**: Tree-based feature importance
2. **Recursive Feature Elimination**: Backward selection with cross-validation
3. **Chi-Square Test**: Statistical significance testing
4. **Combined Scoring**: Multi-method feature ranking

### Model Training
1. **Data Splitting**: 80% train, 20% test with stratification
2. **Cross-Validation**: 5-fold CV for model evaluation
3. **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
4. **Model Persistence**: Joblib serialization for deployment

### Clustering Analysis
1. **K-Means**: Elbow method for optimal cluster selection
2. **Hierarchical**: Multiple linkage methods comparison
3. **Evaluation**: Silhouette score, ARI, NMI metrics
4. **Visualization**: PCA-based cluster plotting

## 🚀 Deployment

### Local Deployment
```bash
streamlit run ui/app.py
```

### Ngrok Deployment (Optional)
```bash
# Install ngrok
# Download from https://ngrok.com/

# Run the Streamlit app
streamlit run ui/app.py

# In another terminal, expose the app
ngrok http 8501
```

## 📊 Results and Visualizations

The pipeline generates comprehensive visualizations:
- EDA analysis with distribution plots
- PCA variance analysis and component plots
- Feature importance rankings
- Model performance comparisons
- Clustering visualizations
- Hyperparameter tuning results

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 📞 Contact

For questions or support, please refer to the project documentation or contact the development team.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Active Development
