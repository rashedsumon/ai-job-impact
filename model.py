import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_loader import download_and_load_dataset

MODEL_FILE = "ai_impact_model.joblib"

def build_and_train_pipeline():
    """
    Downloads data, handles preprocessing, trains a machine learning pipeline, 
    and saves the finalized asset locally.
    """
    # 1. Fetch data
    df = download_and_load_dataset()
    
    # Ensure column casing maps safely. This generalizes common keys present in the data.
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Identify expected columns (Using safe fallbacks if precise string cases vary)
    target_col = 'salary' if 'salary' in df.columns else [c for c in df.columns if 'salary' in c][0]
    industry_col = 'industry' if 'industry' in df.columns else [c for c in df.columns if 'industry' in c][0]
    
    # Fallback search for structural feature tracking automation risk/intensity
    risk_col = [c for c in df.columns if 'risk' in c or 'intensity' in c or 'score' in c]
    risk_col = risk_col[0] if risk_col else df.columns[0] 

    # 2. Select Features and Target
    X = df[[industry_col, risk_col]]
    y = df[target_col]
    
    # 3. Define preprocessing pipelines
    categorical_features = [industry_col]
    numerical_features = [risk_col]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Leave numerical factors alone
    )
    
    # 4. Construct complete ML pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # 5. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training the Random Forest model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate score
    score = model_pipeline.score(X_test, y_test)
    print(f"Model R^2 evaluation score on test set: {score:.4f}")
    
    # 6. Save Model Artifact
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"Model saved successfully to {MODEL_FILE}")
    return model_pipeline

def get_or_train_model():
    """
    Checks if a trained model exists; if not, triggers training automatically.
    """
    if os.path.exists(MODEL_FILE):
        print("Loading existing trained model pipeline...")
        return joblib.load(MODEL_FILE)
    else:
        print("No pre-existing model found. Training now...")
        return build_and_train_pipeline()

if __name__ == "__main__":
    build_and_train_pipeline()