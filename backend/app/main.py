"""
Credit Risk Platform
Day 16 - Production-Style FastAPI Backend

Features:
- Loan default risk prediction
- Random Forest model
- Feature engineering
- Prediction history
- Prediction statistics
- Health check
- API version information
- Model information
- CORS support
- Input validation
"""
from sqlalchemy.orm import Session
from fastapi import Depends

from .database import Base, engine, get_db
from .models import LoanApplication as LoanApplicationDB
from .models import Prediction

import json
import os
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

HISTORY_PATH = "backend/app/prediction_history.json"

API_VERSION = "1.4.0"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "AI-powered credit risk prediction using "
        "a Random Forest model."
    ),
    version=API_VERSION,
)
Base.metadata.create_all(bind=engine)


# ============================================================
# CORS
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
# Prediction History File
# ============================================================

def ensure_history_file():

    directory = os.path.dirname(HISTORY_PATH)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    if not os.path.exists(HISTORY_PATH):

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


ensure_history_file()


# ============================================================
# Request Schema
# ============================================================

class LoanApplication(BaseModel):

    loan_amnt: float = Field(
        gt=0,
        description="Requested loan amount"
    )

    term: str = Field(
        min_length=1,
        description="Loan term, e.g. 36 months"
    )

    int_rate: float = Field(
        gt=0,
        description="Interest rate percentage"
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

    emp_length: str = Field(
        min_length=1,
        description="Employment length"
    )

    home_ownership: str = Field(
        min_length=1,
        description="Home ownership"
    )

    annual_inc: float = Field(
        ge=0,
        description="Annual income"
    )

    verification_status: str = Field(
        min_length=1,
        description="Income verification status"
    )

    purpose: str = Field(
        min_length=1,
        description="Loan purpose"
    )

    dti: float = Field(
        ge=0,
        description="Debt-to-income ratio"
    )

    delinq_2yrs: float = Field(
        ge=0,
        description="Delinquencies during last 2 years"
    )

    open_acc: float = Field(
        ge=0,
        description="Number of open accounts"
    )

    pub_rec: float = Field(
        ge=0,
        description="Public records"
    )

    revol_bal: float = Field(
        ge=0,
        description="Revolving balance"
    )

    revol_util: float = Field(
        ge=0,
        le=100,
        description="Revolving utilization percentage"
    )

    total_acc: float = Field(
        ge=0,
        description="Total accounts"
    )

    application_type: str = Field(
        min_length=1,
        description="Application type"
    )


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "healthy",
        "version": API_VERSION
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    model_exists = os.path.exists(
        MODEL_PATH
    )

    return {
        "status": (
            "healthy"
            if model_exists
            else "unhealthy"
        ),
        "model_loaded": model_exists,
        "model_type": "Random Forest",
        "api_version": API_VERSION
    }


# ============================================================
# API Version
# ============================================================

@app.get("/api/version")
def api_version():

    return {
        "name": "Credit Risk Prediction API",
        "version": API_VERSION,
        "status": "stable",
        "model": "Random Forest",
        "purpose": "Loan default risk assessment"
    }


# ============================================================
# Model Information
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Random Forest",
        "task": "Binary Classification",
        "purpose": "Credit Risk Prediction",

        "metrics": {
            "roc_auc": 0.7096,
            "accuracy": 0.6385,
            "precision": 0.3284,
            "recall": 0.6721,
            "f1_score": 0.4412
        },

        "classes": {
            "0": "Low Risk",
            "1": "High Risk"
        }
    }


# ============================================================
# Feature Engineering
# ============================================================

def prepare_features(
    application: LoanApplication
):

    # --------------------------------------------------------
    # Convert request to DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame(
        [
            application.model_dump()
        ]
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
        .str.extract(
            r"(\d+)"
        )[0]
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

        "10+ years": 10

    }


    data["emp_length"] = (
        data["emp_length"]
        .map(emp_map)
    )


    # --------------------------------------------------------
    # Validate employment length
    # --------------------------------------------------------

    if data["emp_length"].isna().any():

        raise ValueError(
            "Invalid employment length value."
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
    # INCOME
    # --------------------------------------------------------

    annual_income = (
        data["annual_inc"]
        .replace(
            0,
            float("nan")
        )
    )


    # --------------------------------------------------------
    # LOAN TO INCOME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Replace invalid infinite values
    # --------------------------------------------------------

    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )


    return data


# ============================================================
# Save Prediction History
# ============================================================

def save_prediction_history(
    prediction: int,
    risk: str,
    probability: float
):

    ensure_history_file()


    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)


    except (
        json.JSONDecodeError,
        FileNotFoundError
    ):

        history = []


    record = {

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "prediction":
            int(prediction),

        "risk":
            risk,

        "default_probability":
            round(
                float(probability),
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


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(
    application: LoanApplication
):

    try:

        # ----------------------------------------------------
        # Prepare features
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


        prediction = int(
            prediction
        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(data)[0]
        )


        probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # Protect probability range
        # ----------------------------------------------------

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )


        # ----------------------------------------------------
        # Risk Classification
        # ----------------------------------------------------

        if prediction == 1:

            risk = "High Risk"

        else:

            risk = "Low Risk"


        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        save_prediction_history(
            prediction,
            risk,
            probability
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "prediction":
                prediction,

            "risk":
                risk,

            "default_probability":
                round(
                    probability,
                    4
                ),

            "default_probability_percent":
                round(
                    probability * 100,
                    2
                )

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
def get_history():

    ensure_history_file()


    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)


    except (
        json.JSONDecodeError,
        FileNotFoundError
    ):

        history = []


    return {

        "count":
            len(history),

        "history":
            history

    }


# ============================================================
# Clear Prediction History
# ============================================================

@app.delete("/history")
def clear_history():

    ensure_history_file()


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


    return {

        "message":
            "Prediction history cleared successfully",

        "count":
            0

    }


# ============================================================
# Prediction Statistics
# ============================================================

@app.get("/stats")
def get_statistics():

    ensure_history_file()


    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)


    except (
        json.JSONDecodeError,
        FileNotFoundError
    ):

        history = []


    # --------------------------------------------------------
    # Empty history
    # --------------------------------------------------------

    if not history:

        return {

            "total_predictions":
                0,

            "low_risk":
                0,

            "high_risk":
                0,

            "average_default_probability":
                0,

            "high_risk_percentage":
                0

        }


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_predictions = len(
        history
    )


    high_risk = sum(
        1
        for item in history
        if item.get("prediction") == 1
    )


    low_risk = (
        total_predictions
        - high_risk
    )


    probabilities = [

        float(
            item.get(
                "default_probability",
                0
            )
        )

        for item in history

    ]


    average_probability = (
        sum(probabilities)
        / len(probabilities)
    )


    high_risk_percentage = (
        high_risk
        / total_predictions
    ) * 100


    return {

        "total_predictions":
            total_predictions,

        "low_risk":
            low_risk,

        "high_risk":
            high_risk,

        "average_default_probability":
            round(
                average_probability,
                4
            ),

        "high_risk_percentage":
            round(
                high_risk_percentage,
                2
            )

    }