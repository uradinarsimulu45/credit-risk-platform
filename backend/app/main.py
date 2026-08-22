"""
Credit Risk Platform
Day 18 - FastAPI + ML Model + MySQL
"""

import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.database import (
    save_prediction,
    get_prediction_history,
    clear_prediction_history,
    get_prediction_stats,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "ml/models/best_credit_risk_model.joblib"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "AI-powered credit risk prediction using "
        "a Random Forest model with MySQL prediction history."
    ),
    version="1.4.0",
)


# ============================================================
# CORS Configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load Machine Learning Model
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

    loan_amnt: float = Field(
        gt=0,
        description="Requested loan amount"
    )

    term: str = Field(
        description="Loan term, e.g. 36 months"
    )

    int_rate: float = Field(
        gt=0,
        description="Interest rate"
    )

    installment: float = Field(
        gt=0,
        description="Monthly installment"
    )

    grade: str = Field(
        min_length=1,
        max_length=1,
        description="Loan grade A-G"
    )

    sub_grade: str = Field(
        min_length=2,
        max_length=2,
        description="Loan sub-grade, e.g. B3"
    )

    emp_length: str

    home_ownership: str

    annual_inc: float = Field(
        ge=0,
        description="Annual income"
    )

    verification_status: str

    purpose: str

    dti: float = Field(
        ge=0,
        description="Debt-to-income ratio"
    )

    delinq_2yrs: float = Field(
        ge=0
    )

    open_acc: float = Field(
        ge=0
    )

    pub_rec: float = Field(
        ge=0
    )

    revol_bal: float = Field(
        ge=0
    )

    revol_util: float = Field(
        ge=0,
        le=100,
        description="Revolving utilization percentage"
    )

    total_acc: float = Field(
        ge=0
    )

    application_type: str


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "healthy",
        "version": "1.4.0",
        "database": "MySQL",
        "model": "Random Forest",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Random Forest",
        "database": "MySQL",
    }


# ============================================================
# Model Information
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Random Forest",
        "purpose": "Credit Risk Prediction",
        "roc_auc": 0.7096,
        "accuracy": 0.6385,
        "precision": 0.3284,
        "recall": 0.6721,
        "f1_score": 0.4412,
    }


# ============================================================
# Feature Engineering
# ============================================================

def prepare_features(
    application: LoanApplication
):

    # Convert request into DataFrame

    data = pd.DataFrame(
        [application.model_dump()]
    )


    # --------------------------------------------------------
    # TERM
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
    # EMPLOYMENT LENGTH
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
    # REVOLVING UTILIZATION
    # --------------------------------------------------------

    data["revol_util"] = (
        data["revol_util"]
        .astype(str)
        .str.replace(
            "%",
            "",
            regex=False
        )
        .astype(float)
    )


    # --------------------------------------------------------
    # LOAN TO INCOME
    # --------------------------------------------------------

    annual_income = (
        data["annual_inc"]
        .replace(
            0,
            float("nan")
        )
    )


    data["loan_to_income"] = (
        data["loan_amnt"]
        / annual_income
    )


    # --------------------------------------------------------
    # INSTALLMENT TO INCOME
    # --------------------------------------------------------

    data["installment_to_income"] = (
        (data["installment"] * 12)
        / annual_income
    )


    # --------------------------------------------------------
    # INCOME MISSING
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
def predict(
    application: LoanApplication
):

    try:

        # ----------------------------------------------------
        # Prepare Features
        # ----------------------------------------------------

        data = prepare_features(
            application
        )


        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            data
        )[0]


        # ----------------------------------------------------
        # Default Probability
        # ----------------------------------------------------

        probability = (
            model.predict_proba(data)[0][1]
        )


        # ----------------------------------------------------
        # Risk Classification
        # ----------------------------------------------------

        if prediction == 1:

            risk = "High Risk"

        else:

            risk = "Low Risk"


        # ----------------------------------------------------
        # Save Prediction to MySQL
        # ----------------------------------------------------

        prediction_id = save_prediction(

            prediction=int(
                prediction
            ),

            risk=risk,

            default_probability=float(
                probability
            ),
        )


        # ----------------------------------------------------
        # API Response
        # ----------------------------------------------------

        return {

            "prediction": int(
                prediction
            ),

            "risk": risk,

            "default_probability": round(
                float(probability),
                4
            ),

            "prediction_id": prediction_id,

            "message": (
                "Prediction generated and "
                "saved to MySQL successfully."
            ),
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# Prediction History
# ============================================================

@app.get("/history")
def history():

    try:

        records = (
            get_prediction_history()
        )


        return {

            "count": len(records),

            "history": records,
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# Clear Prediction History
# ============================================================

@app.delete("/history")
def clear_history():

    try:

        deleted_records = (
            clear_prediction_history()
        )


        return {

            "message": (
                "Prediction history "
                "cleared successfully."
            ),

            "deleted_records": deleted_records,
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# Prediction Statistics
# ============================================================

@app.get("/stats")
def stats():

    try:

        return get_prediction_stats()


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )