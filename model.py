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
    Loads raw dataset columns matching the user's schema index, preprocesses categorical values,
    and saves an automated pipeline artifact.
    """
    # 1. Fetch data via dataset manager script
    df = download_and_load_dataset()
    
    # Clean whitespace edge cases around column strings just in case
    df.columns = [col.strip() for col in df.columns]
    
    # 2. Select precise features and target matching your schema index
    feature_cols = ['industry', 'ai_intensity_score', 'automation_risk_score']
    target_col = 'salary_usd'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 3. Setup structural transformations
    categorical_features = ['industry']
    numerical_features = ['ai_intensity_score', 'automation_risk_score']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Passes numerical features directly without alterations
    )
    
    # 4. Create complete modular ML pipeline structure
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 5. Segment, fit and assess performance metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Random Forest targeting salary forecasting...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate score accuracy output
    score = model_pipeline.score(X_test, y_test)
    print(f"Pipeline successfully compiled. R^2 Validation Score: {score:.4f}")
    
    # 6. Save binary operational asset
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"Model saved successfully as {MODEL_FILE}")
    return model_pipeline

def get_or_train_model():
    """
    Checks if a trained model exists; if not, triggers training automatically.
    """
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return build_and_train_pipeline()

if __name__ == "__main__":
    build_and_train_pipeline()