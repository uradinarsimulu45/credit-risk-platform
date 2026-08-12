"""
Day 7 - Credit Risk Prediction API

FastAPI backend for the Credit Risk Platform.
"""

import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "ml/models/best_credit_risk_model.joblib"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting loan default risk",
    version="1.0.0",
)


# ============================================================
# Load Trained Model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# Request Schema
# ============================================================

class LoanApplication(BaseModel):

    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    delinq_2yrs: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    application_type: str


# ============================================================
# Root / Health Check
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "healthy",
        "model": "Random Forest",
    }


# ============================================================
# Feature Engineering
# ============================================================

def prepare_features(application: LoanApplication):

    # Convert request to DataFrame
    data = pd.DataFrame(
        [application.model_dump()]
    )

    # --------------------------------------------------------
    # Convert TERM
    #
    # "36 months" -> 36
    # "60 months" -> 60
    # --------------------------------------------------------

    data["term"] = (
        data["term"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    # --------------------------------------------------------
    # Convert EMPLOYMENT LENGTH
    #
    # "< 1 year" -> 0
    # "1 year"   -> 1
    # "5 years"  -> 5
    # "10+ years" -> 10
    # --------------------------------------------------------

    emp_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }

    data["emp_length"] = (
        data["emp_length"]
        .map(emp_map)
    )

    # --------------------------------------------------------
    # Convert REVOL UTIL
    #
    # "45.5%" -> 45.5
    # --------------------------------------------------------

    data["revol_util"] = (
        data["revol_util"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # --------------------------------------------------------
    # Loan-to-Income
    # --------------------------------------------------------

    data["loan_to_income"] = (
        data["loan_amnt"]
        / data["annual_inc"].replace(0, float("nan"))
    )

    # --------------------------------------------------------
    # Installment-to-Income
    # --------------------------------------------------------

    data["installment_to_income"] = (
        (data["installment"] * 12)
        / data["annual_inc"].replace(0, float("nan"))
    )

    # --------------------------------------------------------
    # Income Missing Flag
    # --------------------------------------------------------

    data["income_missing"] = (
        data["annual_inc"]
        .isna()
        .astype(int)
    )

    return data


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    try:

        # Prepare features exactly as during training
        data = prepare_features(application)

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1]

        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        if prediction == 1:
            risk = "High Risk"
        else:
            risk = "Low Risk"

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "prediction": int(prediction),
            "risk": risk,
            "default_probability": round(
                float(probability),
                4
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )