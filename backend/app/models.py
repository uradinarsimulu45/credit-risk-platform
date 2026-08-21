from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)

from .database import Base


# ============================================================
# Loan Application Table
# ============================================================

class LoanApplication(Base):

    __tablename__ = "loan_applications"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    loan_amnt = Column(Float, nullable=False)

    term = Column(String(20), nullable=False)

    int_rate = Column(Float, nullable=False)

    installment = Column(Float, nullable=False)

    grade = Column(String(5), nullable=False)

    sub_grade = Column(String(5), nullable=False)

    emp_length = Column(String(30), nullable=False)

    home_ownership = Column(String(30), nullable=False)

    annual_inc = Column(Float, nullable=False)

    verification_status = Column(
        String(50),
        nullable=False,
    )

    purpose = Column(
        String(100),
        nullable=False,
    )

    dti = Column(Float, nullable=False)

    delinq_2yrs = Column(Float, nullable=False)

    open_acc = Column(Float, nullable=False)

    pub_rec = Column(Float, nullable=False)

    revol_bal = Column(Float, nullable=False)

    revol_util = Column(Float, nullable=False)

    total_acc = Column(Float, nullable=False)

    application_type = Column(
        String(50),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# Prediction Table
# ============================================================

class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    application_id = Column(
        Integer,
        ForeignKey(
            "loan_applications.id"
        ),
        nullable=False,
    )

    prediction = Column(
        Integer,
        nullable=False,
    )

    risk = Column(
        String(30),
        nullable=False,
    )

    default_probability = Column(
        Float,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )