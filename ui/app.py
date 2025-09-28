"""
Streamlit Web UI for Heart Disease Prediction
Real-time prediction interface with data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .safe-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-prediction {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models"""
    try:
        # Try to load the best tuned model first
        best_model = joblib.load('models/best_tuned_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return best_model, scaler, "Best Tuned Model"
    except FileNotFoundError:
        try:
            # Fallback to basic models
            best_model = joblib.load('models/best_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            return best_model, scaler, "Best Model"
        except FileNotFoundError:
            st.error("Models not found. Please run the training pipeline first.")
            return None, None, None

@st.cache_data
def load_data():
    """Load processed data for visualization"""
    try:
        df = pd.read_csv('data/heart_disease_processed.csv')
        return df
    except FileNotFoundError:
        return None

def create_input_form():
    """Create input form for user data"""
    st.markdown('<div class="sub-header">Patient Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=20, max_value=100, value=50, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Patient's gender")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                         format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                                              3: "Non-anginal Pain", 4: "Asymptomatic"}[x],
                         help="Type of chest pain experienced")
        trestbps = st.slider("Resting Blood Pressure", min_value=80, max_value=250, value=120, 
                            help="Resting blood pressure in mm Hg")
        chol = st.slider("Serum Cholesterol", min_value=100, max_value=600, value=200, 
                        help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes", 
                          help="Fasting blood sugar > 120 mg/dl")
    
    with col2:
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                              format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 
                                                   2: "Left Ventricular Hypertrophy"}[x],
                              help="Resting electrocardiographic results")
        thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, 
                           help="Maximum heart rate achieved during exercise")
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes", 
                            help="Exercise induced angina")
        oldpeak = st.slider("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1, 
                           help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3], 
                            format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
                            help="Slope of the peak exercise ST segment")
        ca = st.slider("Number of Major Vessels", min_value=0, max_value=3, value=0, 
                      help="Number of major vessels colored by flourosopy")
        thal = st.selectbox("Thalassemia", options=[3, 6, 7], 
                           format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x],
                           help="Thalassemia type")
    
    return {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }

def make_prediction(model, scaler, input_data):
    """Make prediction using the trained model"""
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability

def display_prediction(prediction, probability):
    """Display prediction results"""
    st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
    
    # Calculate confidence
    confidence = max(probability) * 100
    
    # Create prediction card
    if prediction == 0:
        card_class = "prediction-card safe-prediction"
        result_text = "NO HEART DISEASE"
        result_icon = "✅"
        risk_level = "Low Risk"
    else:
        card_class = "prediction-card risk-prediction"
        result_text = "HEART DISEASE DETECTED"
        result_icon = "⚠️"
        risk_level = "High Risk"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h2>{result_icon} {result_text}</h2>
        <h3>Risk Level: {risk_level}</h3>
        <h4>Confidence: {confidence:.1f}%</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("No Heart Disease", f"{probability[0]*100:.1f}%")
    with col2:
        st.metric("Heart Disease", f"{probability[1]*100:.1f}%")
    
    # Risk interpretation
    if prediction == 0:
        st.success("🎉 Great news! Based on the provided information, the model predicts no heart disease. However, please consult with a healthcare professional for a comprehensive evaluation.")
    else:
        st.warning("⚠️ The model indicates a potential risk of heart disease. Please consult with a healthcare professional immediately for further evaluation and appropriate care.")

def create_data_visualizations(df):
    """Create data visualizations"""
    st.markdown('<div class="sub-header">Data Analysis & Insights</div>', unsafe_allow_html=True)
    
    if df is None:
        st.warning("Data not available for visualization.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "❤️ Heart Disease Analysis", "📈 Feature Analysis", "🔍 Interactive Plots"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Heart Disease Cases", df['target'].sum())
        with col3:
            st.metric("No Heart Disease", len(df) - df['target'].sum())
        with col4:
            st.metric("Disease Rate", f"{df['target'].mean()*100:.1f}%")
        
        # Age distribution
        fig_age = px.histogram(df, x='age', color='target', 
                              title="Age Distribution by Heart Disease Status",
                              labels={'age': 'Age', 'count': 'Number of Patients'},
                              color_discrete_map={0: 'lightblue', 1: 'red'})
        st.plotly_chart(fig_age, use_container_width=True)
    
    with tab2:
        st.subheader("Heart Disease Analysis")
        
        # Disease distribution
        disease_counts = df['target'].value_counts()
        fig_pie = px.pie(values=disease_counts.values, names=['No Heart Disease', 'Heart Disease'],
                        title="Heart Disease Distribution",
                        color_discrete_map={0: 'lightblue', 1: 'red'})
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Gender analysis
        gender_analysis = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
        fig_gender = px.bar(gender_analysis, title="Heart Disease by Gender",
                           labels={'sex': 'Gender', 'value': 'Number of Patients'})
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Analysis")
        
        # Correlation heatmap
        feature_cols = [col for col in df.columns if col not in ['target', 'dataset']]
        correlation_matrix = df[feature_cols + ['target']].corr()
        
        fig_corr = px.imshow(correlation_matrix, 
                            title="Feature Correlation Heatmap",
                            color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature importance (if available)
        try:
            # Try to load feature importance from results
            feature_importance = pd.read_csv('../results/feature_selection_summary.csv')
            if 'rf_importance' in feature_importance.columns:
                fig_importance = px.bar(feature_importance.head(10), 
                                      x='rf_importance', y='feature',
                                      title="Top 10 Most Important Features",
                                      orientation='h')
                st.plotly_chart(fig_importance, use_container_width=True)
        except FileNotFoundError:
            st.info("Feature importance data not available. Run the feature selection pipeline first.")
    
    with tab4:
        st.subheader("Interactive Data Exploration")
        
        # Feature selection for plotting
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", feature_cols)
        with col2:
            y_feature = st.selectbox("Y-axis Feature", feature_cols)
        
        # Create scatter plot
        fig_scatter = px.scatter(df, x=x_feature, y=y_feature, color='target',
                               title=f"{x_feature} vs {y_feature}",
                               labels={'target': 'Heart Disease'},
                               color_discrete_map={0: 'lightblue', 1: 'red'})
        st.plotly_chart(fig_scatter, use_container_width=True)

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models and data
    model, scaler, model_name = load_models()
    df = load_data()
    
    if model is None:
        st.error("Please run the training pipeline first to generate the required models.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🔮 Prediction", "📊 Data Analysis", "ℹ️ About"])
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Heart Disease Prediction System
        
        This application uses machine learning to predict the likelihood of heart disease based on various health parameters.
        
        ### Features:
        - **Real-time Prediction**: Get instant heart disease risk assessment
        - **Data Visualization**: Explore heart disease patterns and trends
        - **Interactive Analysis**: Customize your data exploration
        - **Comprehensive Metrics**: Detailed performance analysis
        
        ### How to Use:
        1. Navigate to the **Prediction** page
        2. Enter patient information using the sliders and dropdowns
        3. Click "Predict" to get the risk assessment
        4. Explore the **Data Analysis** page for insights
        
        ### Model Information:
        - **Model Type**: {model_name}
        - **Features Used**: 13 health parameters
        - **Accuracy**: Optimized through hyperparameter tuning
        
        ### Disclaimer:
        This tool is for educational and research purposes only. Always consult with healthcare professionals for medical decisions.
        """.format(model_name=model_name))
    
    elif page == "🔮 Prediction":
        st.markdown("### Enter Patient Information")
        
        # Create input form
        input_data = create_input_form()
        
        # Prediction button
        if st.button("🔮 Predict Heart Disease Risk", type="primary"):
            with st.spinner("Analyzing patient data..."):
                prediction, probability = make_prediction(model, scaler, input_data)
                display_prediction(prediction, probability)
        
        # Show input summary
        st.markdown("### Input Summary")
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df, use_container_width=True)
    
    elif page == "📊 Data Analysis":
        create_data_visualizations(df)
    
    elif page == "ℹ️ About":
        st.markdown("""
        ## About the Heart Disease Prediction System
        
        ### Project Overview
        This system is part of a comprehensive machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset.
        
        ### Technical Details
        - **Dataset**: UCI Heart Disease Dataset (Cleveland, Hungarian, Switzerland, VA)
        - **Preprocessing**: Data cleaning, scaling, feature selection
        - **Models**: Logistic Regression, Decision Tree, Random Forest, SVM
        - **Optimization**: Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
        - **Deployment**: Streamlit web application
        
        ### Features Used
        1. **Age**: Patient's age in years
        2. **Sex**: Gender (0: Female, 1: Male)
        3. **Chest Pain Type**: Type of chest pain experienced
        4. **Resting Blood Pressure**: Resting blood pressure in mm Hg
        5. **Serum Cholesterol**: Serum cholesterol in mg/dl
        6. **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl
        7. **Resting ECG**: Resting electrocardiographic results
        8. **Maximum Heart Rate**: Maximum heart rate achieved
        9. **Exercise Induced Angina**: Exercise induced angina
        10. **ST Depression**: ST depression induced by exercise
        11. **Slope**: Slope of peak exercise ST segment
        12. **Major Vessels**: Number of major vessels colored by flourosopy
        13. **Thalassemia**: Thalassemia type
        
        ### Model Performance
        The system uses the best performing model after comprehensive hyperparameter tuning, achieving high accuracy in heart disease prediction.
        
        ### Contact & Support
        For questions or support, please refer to the project documentation or contact the development team.
        
        ### Version
        Version 1.0 - Initial Release
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Heart Disease Prediction System | Built with Streamlit | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
