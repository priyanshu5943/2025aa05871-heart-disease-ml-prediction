import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.title("Heart Disease Prediction - ML Models")
st.markdown("### 2025aa05872 Machine Learning Assignment ")
st.markdown("---")

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression_model.pkl',
        'Decision Tree': 'model/decision_tree_model.pkl',
        'KNN': 'model/knn_model.pkl',
        'Naive Bayes': 'model/naive_bayes_model.pkl',
        'Random Forest': 'model/random_forest_model.pkl',
        'XGBoost': 'model/xgboost_model.pkl'
    }
    
    for name, filepath in model_files.items():
        with open(filepath, 'rb') as file:
            models[name] = pickle.load(file)
    
    return models

models = load_models()

st.subheader("Select Model")
selected_model_name = st.selectbox("Choose a model:", list(models.keys()))
selected_model = models[selected_model_name]

st.markdown("---")

st.subheader("Upload Test Data")

if os.path.exists('model/test_data_sample.csv'):
    df_sample = pd.read_csv('model/test_data_sample.csv')
    csv_data = df_sample.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Sample Test Data (100 rows)",
        data=csv_data,
        file_name="test_data_sample.csv",
        mime="text/csv",
        help="Download sample data to test the app"
    )
    st.info("💡 Don't have test data? Use the download button above!")
else:
    st.warning("Sample test data not available")

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV file with test data", type=['csv'])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success(f"File uploaded successfully! Rows: {len(test_data)}")
    
    st.write("Data Preview:")
    st.dataframe(test_data.head())
    
    if 'Heart Disease' in test_data.columns:
        X_test = test_data.drop('Heart Disease', axis=1)
        y_test = test_data['Heart Disease']
        
        y_pred = selected_model.predict(X_test)
        y_pred_proba = selected_model.predict_proba(X_test)[:, 1] if hasattr(selected_model, 'predict_proba') else y_pred
        
        st.markdown("---")
        
        st.subheader("Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            st.metric("AUC Score", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
        
        with col2:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        
        with col3:
            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            st.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        st.markdown("---")
        
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Absence', 'Presence'],
                    yticklabels=['Absence', 'Presence'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {selected_model_name}')
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Classification Report")
        
        report = classification_report(y_test, y_pred, 
                                       target_names=['Absence', 'Presence'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
    else:
        st.error("❌ The uploaded file must contain a 'Heart Disease' column!")

else:
    st.info("👆 Please upload a CSV file to begin.")

st.markdown("---")
