"""
Credit Risk Platform
Day 11 - FastAPI Backend + Prediction History
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "ml/models/best_credit_risk_model.joblib"

HISTORY_PATH = os.path.join(
    os.path.dirname(__file__),
    "prediction_history.json"
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Credit risk prediction using a Random Forest model.",
    version="1.3.0",
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
# Load Model
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
# Prediction History Functions
# ============================================================

def load_history():
    """Load prediction history from JSON."""

    if not os.path.exists(HISTORY_PATH):
        return []

    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


def save_prediction(
    prediction,
    risk,
    default_probability
):
    """Save a prediction to history."""

    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prediction": int(prediction),
        "risk": risk,
        "default_probability": round(
            float(default_probability),
            4
        )
    }

    history.append(record)

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    return record


def clear_prediction_history():
    """Clear all prediction history."""

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "healthy",
        "version": "1.3.0",
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

    # Convert request to DataFrame

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

    data["loan_to_income"] = (
        data["loan_amnt"]
        / data["annual_inc"].replace(
            0,
            float("nan")
        )
    )

    # --------------------------------------------------------
    # INSTALLMENT TO INCOME
    # --------------------------------------------------------

    data["installment_to_income"] = (
        (data["installment"] * 12)
        / data["annual_inc"].replace(
            0,
            float("nan")
        )
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
        # Prepare input features
        # ----------------------------------------------------

        data = prepare_features(
            application
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            data
        )[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = model.predict_proba(
            data
        )[0][1]

        # ----------------------------------------------------
        # Risk Classification
        # ----------------------------------------------------

        if prediction == 1:

            risk = "High Risk"

        else:

            risk = "Low Risk"

        # ----------------------------------------------------
        # Save prediction history
        # ----------------------------------------------------

        record = save_prediction(
            prediction=prediction,
            risk=risk,
            default_probability=probability
        )

        # ----------------------------------------------------
        # API Response
        # ----------------------------------------------------

        return {

            "prediction": int(prediction),

            "risk": risk,

            "default_probability": round(
                float(probability),
                4
            ),

            "timestamp": record[
                "timestamp"
            ]

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# Prediction History Endpoint
# ============================================================

@app.get("/history")
def get_prediction_history():

    try:

        history = load_history()

        return {

            "count": len(history),

            "history": history

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
def delete_prediction_history():

    try:

        clear_prediction_history()

        return {

            "message":
                "Prediction history cleared successfully.",

            "count": 0

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
def prediction_stats():

    try:

        history_file = "backend/app/prediction_history.json"

        # No history yet
        if not os.path.exists(history_file):

            return {
                "total_predictions": 0,
                "low_risk": 0,
                "high_risk": 0,
                "average_default_probability": 0.0,
                "high_risk_percentage": 0.0,
            }

        # Load prediction history
        with open(
            history_file,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

        # Empty history
        if not history:

            return {
                "total_predictions": 0,
                "low_risk": 0,
                "high_risk": 0,
                "average_default_probability": 0.0,
                "high_risk_percentage": 0.0,
            }

        total_predictions = len(history)

        high_risk = sum(
            1
            for record in history
            if record.get("prediction") == 1
        )

        low_risk = (
            total_predictions - high_risk
        )

        probabilities = [
            float(
                record.get(
                    "default_probability",
                    0
                )
            )
            for record in history
        ]

        average_probability = (
            sum(probabilities)
            / len(probabilities)
        )

        high_risk_percentage = (
            high_risk
            / total_predictions
            * 100
        )

        return {
            "total_predictions": total_predictions,
            "low_risk": low_risk,
            "high_risk": high_risk,
            "average_default_probability":
                round(
                    average_probability,
                    4
                ),
            "high_risk_percentage":
                round(
                    high_risk_percentage,
                    2
                ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )