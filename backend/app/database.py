import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Database Configuration
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "credit_risk")


# ============================================================
# Database URL
# ============================================================

DATABASE_URL = (
    f"mysql+pymysql://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# SQLAlchemy Engine
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ============================================================
# Test Connection
# ============================================================

def test_connection():

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT DATABASE()")
            )

            database_name = result.scalar()

            print(
                f"Database connection successful: "
                f"{database_name}"
            )

            return True

    except SQLAlchemyError as error:

        print(
            f"Database connection failed: {error}"
        )

        return False


# ============================================================
# Save Prediction
# ============================================================

def save_prediction(
    prediction,
    risk,
    default_probability
):

    query = text("""
        INSERT INTO prediction_history
        (
            timestamp,
            prediction,
            risk,
            default_probability
        )
        VALUES
        (
            NOW(),
            :prediction,
            :risk,
            :default_probability
        )
    """)

    try:

        with engine.begin() as connection:

            result = connection.execute(
                query,
                {
                    "prediction": int(prediction),
                    "risk": risk,
                    "default_probability": float(
                        default_probability
                    ),
                }
            )

            return result.lastrowid

    except SQLAlchemyError as error:

        raise RuntimeError(
            f"Failed to save prediction: {error}"
        )


# ============================================================
# Get Prediction History
# ============================================================

def get_prediction_history():

    query = text("""
        SELECT
            id,
            timestamp,
            prediction,
            risk,
            default_probability
        FROM prediction_history
        ORDER BY timestamp DESC
    """)

    try:

        with engine.connect() as connection:

            result = connection.execute(query)

            records = []

            for row in result.mappings():

                records.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"].isoformat(),
                    "prediction": int(
                        row["prediction"]
                    ),
                    "risk": row["risk"],
                    "default_probability": float(
                        row["default_probability"]
                    ),
                })

            return records

    except SQLAlchemyError as error:

        raise RuntimeError(
            f"Failed to get prediction history: {error}"
        )


# ============================================================
# Clear Prediction History
# ============================================================

def clear_prediction_history():

    query = text("""
        DELETE FROM prediction_history
    """)

    try:

        with engine.begin() as connection:

            result = connection.execute(query)

            return result.rowcount

    except SQLAlchemyError as error:

        raise RuntimeError(
            f"Failed to clear prediction history: {error}"
        )


# ============================================================
# Get Prediction Statistics
# ============================================================

def get_prediction_stats():

    query = text("""
        SELECT
            COUNT(*) AS total_predictions,

            SUM(
                CASE
                    WHEN prediction = 0
                    THEN 1
                    ELSE 0
                END
            ) AS low_risk,

            SUM(
                CASE
                    WHEN prediction = 1
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk,

            AVG(default_probability)
                AS average_default_probability

        FROM prediction_history
    """)

    try:

        with engine.connect() as connection:

            result = connection.execute(query).mappings().first()

            total = result["total_predictions"] or 0
            low_risk = result["low_risk"] or 0
            high_risk = result["high_risk"] or 0
            average_probability = (
                result["average_default_probability"]
                or 0
            )

            high_risk_percentage = (
                (high_risk / total) * 100
                if total > 0
                else 0
            )

            return {
                "total_predictions": int(total),
                "low_risk": int(low_risk),
                "high_risk": int(high_risk),
                "average_default_probability": round(
                    float(average_probability),
                    4
                ),
                "high_risk_percentage": round(
                    float(high_risk_percentage),
                    2
                ),
            }

    except SQLAlchemyError as error:

        raise RuntimeError(
            f"Failed to get prediction statistics: {error}"
        )