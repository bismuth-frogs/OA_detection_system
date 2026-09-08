"""
tabular_model.py
----------------
Trains and executes inference for an Osteoarthritis (OA) clinical risk assessment model.
Uses tabular inputs: age, BMI, sex, joint injury history, occupational strain,
family history, and physical activity level.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_PATH = "./checkpoints/oa_tabular_gbc.joblib"

FEATURES = [
    "age",
    "bmi",
    "sex",                   # 0: Female, 1: Male
    "joint_injury_history",  # 0: No, 1: Yes
    "occupational_strain",   # 0: Low, 1: Medium, 2: High
    "family_history",        # 0: No, 1: Yes
    "activity_level",        # 0: Sedentary, 1: Moderate, 2: High
]

def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic epidemiological data for OA risk."""
    np.random.seed(seed)
    
    age = np.random.randint(20, 85, size=n_samples)
    bmi = np.random.uniform(18.5, 42.0, size=n_samples)
    sex = np.random.binomial(1, 0.48, size=n_samples) # 1 = Male, 0 = Female
    joint_injury = np.random.binomial(1, 0.22, size=n_samples)
    occupational_strain = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.35, 0.15])
    family_history = np.random.binomial(1, 0.30, size=n_samples)
    activity_level = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])
    
    # Calculate log-odds of OA based on established clinical risk weights
    log_odds = (
        -4.5
        + 0.05 * (age - 40)
        + 0.12 * (bmi - 25)
        + 0.8 * joint_injury
        + 0.4 * occupational_strain
        + 0.6 * family_history
        - 0.2 * activity_level
        + np.where(sex == 0, 0.3, 0.0) # Females slightly higher risk post-menopause
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    target = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "sex": sex,
        "joint_injury_history": joint_injury,
        "occupational_strain": occupational_strain,
        "family_history": family_history,
        "activity_level": activity_level,
        "oa_risk": target
    })
    return df

def train_and_save_model(model_path: str = MODEL_PATH):
    """Trains Gradient Boosting model on tabular clinical dataset and saves artifact."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = generate_synthetic_data()
    
    X = df[FEATURES]
    y = df["oa_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    print(f"Tabular model trained and saved to {model_path}")

def load_tabular_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        train_and_save_model(model_path)
    return joblib.load(model_path)

def predict_tabular_risk(model, record: dict) -> dict:
    """Evaluates individual clinical risk inputs."""
    input_df = pd.DataFrame([record])[FEATURES]
    prob = float(model.predict_proba(input_df)[0][1])
    
    if prob >= 0.60:
        risk_category = "High Risk"
    elif prob >= 0.35:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Low Risk"
        
    return {
        "risk_probability": round(prob, 4),
        "risk_category": risk_category,
        "risk_percentage": f"{prob:.1%}"
    }

if __name__ == "__main__":
    train_and_save_model()
