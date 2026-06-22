import streamlit as str
import pandas as pd
from model import get_or_train_model

# Page Configurations
str.set_page_config(page_title="AI Job Impact Predictor", layout="centered")

str.title("🧠 Global AI Impact on Jobs & Salaries")
str.write(
    "Utilize this predictive model to evaluate how industry parameters, AI integration intensity, "
    "and automation risk vectors influence compensation metrics across world markets."
)

@str.cache_resource
def load_cached_pipeline():
    return get_or_train_model()

try:
    # Initialize/Fetch Model Pipeline
    pipeline = load_cached_pipeline()
    
    str.subheader("🔮 Run Compensation Inference Engine")
    
    # User Input Forms Configuration
    industry = str.selectbox(
        "Industry Segment:",
        ["Tech", "Finance", "Healthcare", "Manufacturing", "Education", "Retail", "Energy", "Government"]
    )
    
    ai_intensity = str.slider(
        "AI Intensity Score (Centrality of AI in the role):",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    
    automation_risk = str.slider(
        "Automation Risk Score (Vulnerability matrix evaluation):",
        min_value=0.0, max_value=1.0, value=0.4, step=0.01
    )
    
    # Crucial Fix: Create a DataFrame matching the EXACT feature names used to fit the pipeline
    input_features = ['industry', 'ai_intensity_score', 'automation_risk_score']
    input_data = pd.DataFrame([[industry, ai_intensity, automation_risk]], columns=input_features)
    
    if str.button("Predict Target Market Salary"):
        # Execute prediction pipeline securely with structured column headers
        prediction = pipeline.predict(input_data)[0]
        
        str.success("### Inference Computed Successfully!")
        str.metric(
            label="Predicted Base Salary (USD / Annum)", 
            value=f"${prediction:,.2f}"
        )
        
except Exception as e:
    str.error("An operational error was caught during application runtime instantiation.")
    str.exception(e)