import streamlit as str
import pandas as pd
import numpy as np
from model import get_or_train_model

# Set up page configurations
str.set_page_config(page_title="AI Labor Market Impact Predictor", layout="centered")

str.title("🧠 Global AI Impact on Jobs & Salaries")
str.write(
    "This web application utilizes machine learning to predict expected salaries "
    "based on cross-border industry variables and AI metrics using the `sarcasmos/ai-society` dataset."
)

# Load the model via pipeline cache wrapper
@str.cache_resource
def load_cached_pipeline():
    return get_or_train_model()

try:
    pipeline = load_cached_pipeline()
    
    str.subheader("🔮 Predict Market Salary Dynamics")
    
    # Build Input Forms based on user selections
    industry = str.selectbox(
        "Select Industry Ecosystem:",
        ["Tech", "Finance", "Healthcare", "Manufacturing", "Education", "Retail", "Energy"]
    )
    
    risk_score = str.slider(
        "AI Impact / Automation Intensity Score:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    # Get the expected feature tracking strings from the trained pipeline
    # We pass data dynamically using positional matching structured during training 
    feature_names = ['industry', 'automation risk'] 
    
    input_data = pd.DataFrame([[industry, risk_score]], columns=feature_names)
    
    if str.button("Calculate Projected Salary"):
        # Run prediction pipeline
        prediction = pipeline.predict(input_data)[0]
        
        str.success("### Prediction Generated Successfully!")
        str.metric(
            label="Predicted Market Salary (Normalized USD)", 
            value=f"${prediction:,.2f}"
        )
        
except Exception as e:
    str.error("An error occurred during app execution or dataset initialization.")
    str.exception(e)