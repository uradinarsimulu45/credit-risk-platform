import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODEL LOADING (cached — loads once per process, not per rerun)
# ============================================================

# Adjust this if your folder layout differs. Candidates are tried in order
# so a wrong assumption about __file__'s location doesn't fail silently.
_THIS_FILE = Path(__file__).resolve()
_CANDIDATE_PATHS = [
    _THIS_FILE.parent.parent / "ml" / "models" / "best_credit_risk_model.joblib",
    _THIS_FILE.parent / "ml" / "models" / "best_credit_risk_model.joblib",
    _THIS_FILE.parent.parent.parent / "ml" / "models" / "best_credit_risk_model.joblib",
]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tried = []
    for path in _CANDIDATE_PATHS:
        tried.append(str(path))
        if path.exists():
            try:
                return joblib.load(path), True, None, str(path)
            except Exception as e:
                return None, False, f"Found file at {path} but failed to load it: {e}", str(path)
    return None, False, f"No model file found. Tried:\n" + "\n".join(tried), None


model, model_loaded, model_error, model_path_used = load_model()

# Positive class ("default") index — looked up rather than hardcoded to [1],
# since a model's classes_ order isn't guaranteed to be [0, 1].
def get_positive_class_index(m):
    if hasattr(m, "classes_"):
        classes = list(m.classes_)
        if 1 in classes:
            return classes.index(1)
        if True in classes:
            return classes.index(True)
        # fall back to the last class (conventionally the "positive" one)
        return len(classes) - 1
    return 1

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.12), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(14, 165, 233, 0.10), transparent 30%),
            #f5f7fb;
        color: #172033;
    }
    .block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
    section[data-testid="stSidebar"] * { color: #f8fafc !important; }
    .sidebar-brand { font-size: 25px; font-weight: 800; margin-bottom: 5px; }
    .sidebar-subtitle { color: #94a3b8 !important; font-size: 14px; line-height: 1.5; margin-bottom: 30px; }
    .sidebar-section { border-top: 1px solid #374151; padding-top: 22px; margin-top: 22px; }
    .sidebar-label { font-size: 13px; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .sidebar-value { font-size: 16px; font-weight: 600; color: #f8fafc !important; }
    .model-status { background: rgba(34, 197, 94, 0.14); border: 1px solid rgba(34, 197, 94, 0.35); color: #4ade80 !important; padding: 12px 14px; border-radius: 10px; font-weight: 600; }
    .model-status-error { background: rgba(239, 68, 68, 0.14); border: 1px solid rgba(239, 68, 68, 0.35); color: #fca5a5 !important; padding: 12px 14px; border-radius: 10px; font-weight: 600; font-size: 13px; }
    .hero { background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #1d4ed8 100%); border-radius: 24px; padding: 48px; margin-bottom: 34px; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18); border: 1px solid rgba(255,255,255,0.08); }
    .hero-eyebrow { color: #93c5fd; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 14px; }
    .hero-title { color: #ffffff; font-size: 42px; line-height: 1.1; font-weight: 800; margin-bottom: 15px; }
    .hero-subtitle { color: #cbd5e1; font-size: 17px; line-height: 1.7; max-width: 720px; margin-bottom: 25px; }
    .status { display: inline-block; color: #86efac; background: rgba(34,197,94,0.12); border: 1px solid rgba(134,239,172,0.25); border-radius: 999px; padding: 8px 14px; font-size: 13px; font-weight: 700; letter-spacing: .5px; }
    .status-offline { color: #fca5a5; background: rgba(239,68,68,0.12); border: 1px solid rgba(252,165,165,0.25); }
    .section-title { font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 12px; margin-bottom: 5px; }
    .section-description { color: #64748b; margin-bottom: 22px; }
    .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 25px; margin-bottom: 22px; box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06); }
    .card-title { font-size: 20px; font-weight: 750; color: #0f172a; margin-bottom: 5px; }
    .card-description { color: #64748b; font-size: 14px; margin-bottom: 18px; }
    label { color: #334155 !important; font-weight: 600 !important; }
    div[data-baseweb="input"] { border-radius: 10px !important; }
    div[data-baseweb="select"] { border-radius: 10px !important; }
    .stButton > button { width: 100%; height: 52px; border-radius: 12px; border: none; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; font-size: 16px; font-weight: 700; box-shadow: 0 8px 20px rgba(37,99,235,0.25); }
    .stButton > button:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af); color: white; }
    .result-card { background: #ffffff; border-radius: 20px; padding: 30px; border: 1px solid #e2e8f0; box-shadow: 0 12px 35px rgba(15,23,42,0.08); margin-top: 25px; }
    .result-low { border-left: 6px solid #22c55e; }
    .result-medium { border-left: 6px solid #f59e0b; }
    .result-high { border-left: 6px solid #ef4444; }
    .result-label { color: #64748b; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
    .result-value { font-size: 34px; font-weight: 800; margin-top: 7px; margin-bottom: 20px; color: #0f172a; }
    .metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; box-shadow: 0 6px 20px rgba(15,23,42,0.05); }
    .metric-label { color: #64748b; font-size: 13px; font-weight: 600; }
    .metric-value { color: #0f172a; font-size: 25px; font-weight: 800; margin-top: 5px; }
    .footer { text-align: center; color: #94a3b8; font-size: 13px; margin-top: 50px; padding-top: 25px; border-top: 1px solid #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">📊 Credit Risk</div>
        <div class="sidebar-subtitle">AI-powered loan default risk assessment</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)

    if model_loaded:
        st.markdown('<div class="model-status">● Model Loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="model-status-error">● Model Not Loaded<br>{model_error}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-section">
            <div class="sidebar-label">Platform</div>
            <div class="sidebar-value">Random Forest</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">Purpose</div>
            <div class="sidebar-value">Loan default risk prediction</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-label">Version</div>
            <div class="sidebar-value">1.0.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HERO
# ============================================================

status_class = "status" if model_loaded else "status status-offline"
status_text = "● AI MODEL ONLINE" if model_loaded else "● AI MODEL OFFLINE"

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">Intelligent Lending Platform</div>
        <div class="hero-title">AI-Powered Credit Analytics</div>
        <div class="hero-subtitle">
            Make smarter lending decisions with machine-learning powered credit risk assessment.
        </div>
        <span class="{status_class}">{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE TITLE
# ============================================================

st.markdown('<div class="section-title">Loan Application</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Enter the applicant financial information to evaluate credit default risk.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# LOAN DETAILS
# ============================================================

st.markdown(
    """
    <div class="card">
        <div class="card-title">💰 Loan Details</div>
        <div class="card-description">Basic information about the requested loan.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=1000000.0, value=10000.0, step=500.0)
with col2:
    term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
with col3:
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

# ============================================================
# APPLICANT DETAILS
# ============================================================

st.markdown(
    """
    <div class="card">
        <div class="card-title">👤 Applicant Profile</div>
        <div class="card-description">Employment and income information.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    annual_inc = st.number_input("Annual Income ($)", min_value=0.0, max_value=5000000.0, value=60000.0, step=1000.0)
with col2:
    emp_length = st.selectbox(
        "Employment Length",
        list(range(0, 11)),
        format_func=lambda x: "< 1 year" if x == 0 else "10+ years" if x == 10 else f"{x} years",
    )
with col3:
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"])

# ============================================================
# CREDIT PROFILE
# ============================================================

st.markdown(
    """
    <div class="card">
        <div class="card-title">📈 Credit Profile</div>
        <div class="card-description">Credit history and financial indicators.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"])
with col2:
    # Sub-grade options are derived from the chosen grade so the two fields
    # can never contradict each other (e.g. grade "A" + sub_grade "G5").
    sub_grade_options = [f"{grade}{i}" for i in range(1, 6)]
    sub_grade = st.selectbox("Sub Grade", sub_grade_options)
with col3:
    verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])

col1, col2, col3 = st.columns(3)
with col1:
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
with col2:
    delinq_2yrs = st.number_input("Delinquencies (2 Years)", min_value=0, max_value=50, value=0, step=1)
with col3:
    open_acc = st.number_input("Open Accounts", min_value=0, max_value=100, value=8, step=1)

col1, col2, col3 = st.columns(3)
with col1:
    pub_rec = st.number_input("Public Records", min_value=0, max_value=50, value=0, step=1)
with col2:
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=1000000.0, value=5000.0, step=500.0)
with col3:
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=30.0, step=1.0)

col1, col2 = st.columns(2)
with col1:
    total_acc = st.number_input("Total Accounts", min_value=0, max_value=200, value=15, step=1)
with col2:
    purpose = st.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding", "other",
        ],
    )

application_type = st.selectbox("Application Type", ["Individual", "Joint App"])

# ============================================================
# PREDICTION
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Credit Risk")

if predict_clicked:
    if not model_loaded:
        st.error("Model is not loaded. Check the model path shown in the sidebar.")
        st.stop()

    try:
        term_numeric = int(term)

        income_missing = 1 if annual_inc <= 0 else 0
        safe_income = max(float(annual_inc), 1.0)
        loan_to_income = float(loan_amnt) / safe_income

        monthly_rate = float(int_rate) / 100.0 / 12.0
        if int_rate > 0:
            installment = (
                float(loan_amnt) * monthly_rate
                / (1 - (1 + monthly_rate) ** (-term_numeric))
            )
        else:
            installment = float(loan_amnt) / term_numeric

        installment_to_income = installment / safe_income

        input_data = pd.DataFrame([{
            "loan_amnt": float(loan_amnt),
            "term": float(term_numeric),
            "int_rate": float(int_rate),
            "installment": float(installment),
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": float(emp_length),
            "home_ownership": home_ownership,
            "annual_inc": float(annual_inc),
            "verification_status": verification_status,
            "purpose": purpose,
            "dti": float(dti),
            "delinq_2yrs": float(delinq_2yrs),
            "open_acc": float(open_acc),
            "pub_rec": float(pub_rec),
            "revol_bal": float(revol_bal),
            "revol_util": float(revol_util),
            "total_acc": float(total_acc),
            "application_type": application_type,
            "loan_to_income": float(loan_to_income),
            "installment_to_income": float(installment_to_income),
            "income_missing": int(income_missing),
        }])

        raw_prediction = model.predict(input_data)[0]
        prediction = int(round(float(np.asarray(raw_prediction))))

        if hasattr(model, "predict_proba"):
            pos_idx = get_positive_class_index(model)
            proba_row = model.predict_proba(input_data)[0]
            pos_idx = min(pos_idx, len(proba_row) - 1)
            probability = float(proba_row[pos_idx])
        else:
            probability = float(prediction)

        if probability < 0.30:
            risk, risk_class = "Low Risk", "result-low"
        elif probability < 0.60:
            risk, risk_class = "Medium Risk", "result-medium"
        else:
            risk, risk_class = "High Risk", "result-high"

        # Store everything needed to redraw the result, so it survives
        # reruns triggered by other widgets (e.g. opening the expander below).
        st.session_state.prediction_result = {
            "prediction": prediction,
            "probability": probability,
            "risk": risk,
            "risk_class": risk_class,
            "loan_amnt": loan_amnt,
            "term_numeric": term_numeric,
            "int_rate": int_rate,
            "annual_inc": annual_inc,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "grade": grade,
            "sub_grade": sub_grade,
            "dti": dti,
            "purpose": purpose,
        }

    except Exception as e:
        st.session_state.prediction_result = None
        st.error(f"Prediction failed: {str(e)}")
        st.info(
            "Check that the feature set and dtypes sent to the model match "
            "exactly what it was trained on (column names, order, and encoding)."
        )

# ============================================================
# RESULT DISPLAY (reads from session_state, persists across reruns)
# ============================================================

result = st.session_state.prediction_result
if result:
    probability_percent = result["probability"] * 100

    st.markdown(
        f"""
        <div class="result-card {result['risk_class']}">
            <div class="result-label">Credit Risk Classification</div>
            <div class="result-value">
                {"🟢" if result["risk"] == "Low Risk" else "🟠" if result["risk"] == "Medium Risk" else "🔴"}
                {result["risk"]}
            </div>
            <div class="result-label">Estimated Default Probability</div>
            <div class="result-value">{probability_percent:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Prediction</div>
            <div class="metric-value">{result['prediction']}</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Default Probability</div>
            <div class="metric-value">{probability_percent:.2f}%</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Loan Amount</div>
            <div class="metric-value">${result['loan_amnt']:,.0f}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Risk Probability</div>", unsafe_allow_html=True)
    st.progress(min(max(result["probability"], 0.0), 1.0))

    with st.expander("View Application Details"):
        summary = pd.DataFrame({
            "Field": [
                "Loan Amount", "Loan Term", "Interest Rate", "Annual Income",
                "Employment Length", "Home Ownership", "Credit Grade", "Sub Grade",
                "Debt-to-Income", "Loan Purpose",
            ],
            "Value": [
                f"${result['loan_amnt']:,.2f}",
                f"{result['term_numeric']} months",
                f"{result['int_rate']:.2f}%",
                f"${result['annual_inc']:,.2f}",
                "< 1 year" if result["emp_length"] == 0
                    else "10+ years" if result["emp_length"] == 10
                    else f"{result['emp_length']} years",
                result["home_ownership"],
                result["grade"],
                result["sub_grade"],
                f"{result['dti']:.2f}",
                result["purpose"],
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">Credit Risk Analytics Platform · AI-powered lending decision support</div>
    """,
    unsafe_allow_html=True,
)