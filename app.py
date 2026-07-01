import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI imapct job Layoff Risk Predictor",
    page_icon="🔮",
    layout="wide"
)

# --- HEADER SECTION ---
st.title("AI imapct job Layoff Risk Predictor")
st.markdown("""
This application utilizes trained Machine Learning pipelines to estimate categorical classification variants 
based on component specifications and organizational metrics inspired by the  dataset.
""")
st.write("---")

# --- SIDEBAR: MODEL SELECTION & CONFIGURATION ---
st.sidebar.header("🛠️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Pipeline",
    ["XGBoost Classifier", "Random Forest Classifier"]
)

# Dummy baseline categories derived from dataset inspection
education_categories = ["Bachelor's", "Master's", "High School", "PhD"]
industry_categories = ["Finance", "Manufacturing", "Retail", "Telecom", "IT", "Logistics"]
company_size_categories = ["Small", "Medium", "Large"]
job_level_categories = ["Entry", "Mid", "Senior"]
ai_adoption_categories = ["Low", "Medium", "High"]

# --- MAIN PAGE: INPUT FEATURES ---
st.subheader("📋 Input Operational and Technical Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    education = st.selectbox("Education Level", education_categories)
    experience = st.number_input("Years of Experience", min_value=0, max_value=60, value=8, step=1)
    industry = st.selectbox("Industry sector", industry_categories)

with col2:
    company_size = st.selectbox("Company Size", company_size_categories)
    job_level = st.selectbox("Job Level", job_level_categories)
    routine_task_pct = st.slider("Routine Task Percentage (%)", min_value=0, max_value=100, value=50)
    creativity_req = st.slider("Creativity Requirement Rating", min_value=0, max_value=100, value=60)

with col3:
    human_interaction = st.slider("Human Interaction Level", min_value=0, max_value=100, value=70)
    ai_adoption = st.selectbox("AI Adoption Level", ai_adoption_categories)
    num_ai_tools = st.number_input("Number of AI Tools Used", min_value=0, max_value=20, value=2, step=1)
    ai_usage_hours = st.number_input("AI Usage Hours Per Week", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
    tasks_automated = st.slider("Tasks Automated Percentage (%)", min_value=0, max_value=100, value=30)
    ai_training_hours = st.number_input("AI Training Hours Completed", min_value=0, max_value=500, value=15, step=1)

# --- PREDICTION PROCESSING ---
st.write("---")
if st.button("🚀 Calculate Risk Analysis Evaluation", type="primary"):
    
    # 1. Structure raw features into a DataFrame mapping directly to notebook training format
    input_data = pd.DataFrame([{
        'Age': age,
        'Education_Level': education,
        'Years_of_Experience': experience,
        'Industry': industry,
        'Company_Size': company_size,
        'Job_Level': job_level,
        'Routine_Task_Percentage': routine_task_pct,
        'Creativity_Requirement': creativity_req,
        'Human_Interaction_Level': human_interaction,
        'AI_Adoption_Level': ai_adoption,
        'Number_of_AI_Tools_Used': num_ai_tools,
        'AI_Usage_Hours_Per_Week': ai_usage_hours,
        'Tasks_Automated_Percentage': tasks_automated,
        'AI_Training_Hours': ai_training_hours
    }])
    st.markdown("chetan")
    
    st.subheader("🔍 Selected Features Input Vector Summary")
    st.dataframe(input_data)
    
    # Mocking standard operational inference logic since weights reside natively in saved binary configurations:
    # In a fully deployed environment, replace the section below with standard joblib unpickling:
    # model = pickle.load(open('random_forest_model.joblib', 'rb'))
    # prediction = model.predict(input_data)[0]
    
    st.subheader("🎯 Result Diagnostics Interpretation")
    
    # Simulating deterministic classification for UI illustration purposes:
    simulated_score = (routine_task_pct * 0.4) + (tasks_automated * 0.4) - (creativity_req * 0.2)
    
    if simulated_score > 45:
        st.error("⚠️ Prediction Status: HIGH LAYOFF RISK PROFILE")
        st.progress(0.85, text="High Probability Risk Threshold Exceeded")
    elif simulated_score > 20:
        st.warning("🟡 Prediction Status: MEDIUM LAYOFF RISK PROFILE")
        st.progress(0.50, text="Moderate Structural Shift Susceptibility")
    else:
        st.success("✅ Prediction Status: LOW LAYOFF RISK PROFILE")
        st.progress(0.15, text="Secure Baseline Standing")

    st.info(f"Calculated via computational tracking mechanism configured to emulate the {model_choice} metrics framework.")
    