# 📊 AI-Powered Credit Risk Platform

<p align="center">
  <strong>Machine Learning Based Credit Risk Assessment & Loan Default Prediction</strong>
</p>

<p align="center">
  <a href="https://credit-risk-platform1.streamlit.app/">🚀 Live Demo</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/uradinarsimulu45/credit-risk-platform">💻 GitHub Repository</a>
</p>

---

## 🚀 Live Demo

### 👉 [Open the Credit Risk Platform](https://credit-risk-platform1.streamlit.app/)

The application provides an interactive interface for evaluating loan applications and estimating the probability of loan default using a trained machine-learning model.

---

## 📌 Project Overview

The **Credit Risk Platform** is an AI-powered lending analytics application designed to help evaluate the potential credit risk associated with a loan application.

The platform takes financial, employment, credit-history, and loan information as input and uses a **Random Forest machine-learning model** to classify the application into a risk category.

### Core prediction output

* 🟢 **Low Risk**
* 🟠 **Medium Risk**
* 🔴 **High Risk**
* 📈 Estimated default probability

The project combines **machine learning, Python, Streamlit, FastAPI, SQL, and data analytics** into an end-to-end credit-risk solution.

---

## ✨ Features

### 🤖 Machine Learning

* Random Forest classification
* Credit default risk prediction
* Default probability estimation
* Feature preprocessing
* Engineered financial features
* Model persistence using Joblib

### 📊 Credit Analytics

* Loan risk classification
* Default probability
* Loan-to-income analysis
* Installment-to-income analysis
* Income-missing indicator
* Credit profile analysis

### 🖥️ Professional Web Interface

* Modern Streamlit dashboard
* Responsive layout
* Professional loan application form
* Risk classification cards
* Model status indicator
* Interactive prediction workflow

### 📚 Prediction Analytics

* Prediction history
* Risk statistics
* Risk distribution
* Default probability analysis
* Interactive charts

### 🔌 Backend API

The project also contains a FastAPI backend with endpoints for:

* Health checking
* Model information
* Credit risk prediction
* Prediction history
* Prediction statistics

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User / Lender   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Frontend  │
                    │  Credit Dashboard   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Feature Engineering│
                    │                     │
                    │ loan_to_income      │
                    │ installment_to_income│
                    │ income_missing      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest Model │
                    │                     │
                    │ Credit Risk         │
                    │ Classification      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌──────────────┐     ┌──────────────┐
             │ Risk Result  │     │ Probability  │
             │ Low/Med/High │     │    Score     │
             └──────────────┘     └──────────────┘
```

---

## 🧠 Machine Learning Pipeline

The prediction pipeline follows these major stages:

```text
Loan Application
       ↓
Data Validation
       ↓
Feature Engineering
       ↓
Categorical / Numerical Processing
       ↓
Random Forest Model
       ↓
Prediction
       ↓
Default Probability
       ↓
Risk Classification
```

### Engineered Features

The model uses additional financial indicators including:

| Feature                 | Description                             |
| ----------------------- | --------------------------------------- |
| `loan_to_income`        | Loan amount relative to annual income   |
| `installment_to_income` | Monthly installment relative to income  |
| `income_missing`        | Indicates missing or unavailable income |
| `sub_grade`             | More granular credit-grade information  |

---

## 📥 Input Features

The platform accepts information such as:

### Loan Information

* Loan amount
* Loan term
* Interest rate
* Installment
* Loan purpose

### Applicant Information

* Annual income
* Employment length
* Home ownership
* Income verification status

### Credit Information

* Grade
* Sub-grade
* Debt-to-income ratio
* Delinquencies
* Open accounts
* Public records
* Revolving balance
* Revolving utilization
* Total accounts
* Application type

---

## 📤 Example Prediction

A submitted application produces a result similar to:

```text
Credit Risk Classification
--------------------------
🟢 Low Risk

Estimated Default Probability
-----------------------------
27.43%
```

The probability is generated by the trained machine-learning model.

---

## 🛠️ Technology Stack

### Frontend

* Python
* Streamlit
* HTML
* CSS

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Joblib

### Database

* MySQL
* SQLAlchemy
* PyMySQL

### Data Processing

* PyArrow
* Pandas

### Visualization

* Matplotlib
* Streamlit charts

### Deployment

* Streamlit Community Cloud
* Render

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
credit-risk-platform/
│
├── backend/
│   └── app/
│       ├── main.py
│       └── database.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   └── app.py
│
├── ml/
│   ├── models/
│   │   └── best_credit_risk_model.joblib
│   │
│   ├── data_analysis.py
│   └── ...
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/uradinarsimulu45/credit-risk-platform.git
```

### 2. Navigate to the project

```bash
cd credit-risk-platform
```

### 3. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

### 4. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

```powershell
python -m streamlit run frontend\app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🔌 Run the FastAPI Backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔗 API Endpoints

The FastAPI backend provides endpoints including:

| Method | Endpoint      | Purpose                  |
| ------ | ------------- | ------------------------ |
| GET    | `/`           | API information          |
| GET    | `/health`     | Health check             |
| GET    | `/model-info` | Model information        |
| POST   | `/predict`    | Credit risk prediction   |
| GET    | `/history`    | Prediction history       |
| DELETE | `/history`    | Clear prediction history |
| GET    | `/stats`      | Prediction statistics    |

---

## 🗄️ Database

The backend supports MySQL-based prediction history.

The prediction history table stores:

```text
id
timestamp
prediction
risk
default_probability
```

Example:

```text
+----+---------------------+------------+-----------+---------------------+
| id | timestamp           | prediction | risk      | default_probability |
+----+---------------------+------------+-----------+---------------------+
| 10 | 2026-08-23 12:50:38 | 1          | High Risk | 0.5500              |
|  9 | 2026-08-23 12:50:36 | 1          | High Risk | 0.5500              |
|  8 | 2026-08-23 12:50:35 | 1          | High Risk | 0.5500              |
+----+---------------------+------------+-----------+---------------------+
```

---

## 📊 Analytics Example

The platform can calculate statistics such as:

```text
Total Predictions       : 10
Low Risk                : 7
High Risk               : 3
Average Default Risk    : 39.04%
High Risk Percentage    : 30%
```

These analytics help provide a quick overview of the application's credit-risk distribution.

---

## 🔐 Environment Variables

For database-enabled deployments, configure environment variables rather than committing credentials to GitHub.

Example:

```text
DB_HOST=
DB_PORT=
DB_USER=
DB_PASSWORD=
DB_NAME=
```

Never commit:

```text
.env
database passwords
API keys
private credentials
```

---

## 🚀 Deployment

### Streamlit

The current live application is deployed using Streamlit Community Cloud.

**Live application:**

👉 https://credit-risk-platform1.streamlit.app/

### FastAPI

The backend API can be deployed separately using a cloud hosting provider such as Render.

---

## 🧪 Testing

Before deployment, verify:

```powershell
python -m streamlit run frontend\app.py
```

and test:

* Application loads
* Model loads successfully
* Loan form accepts input
* Prediction executes
* Risk classification is displayed
* Default probability is displayed
* Prediction history works
* Charts render correctly

---

## 🎯 Project Goals

The main objectives of this project are:

1. Build an end-to-end credit risk prediction system.
2. Apply machine learning to lending decisions.
3. Engineer meaningful financial features.
4. Provide an intuitive analytics dashboard.
5. Store and analyze prediction history.
6. Expose prediction functionality through an API.
7. Deploy the application to the cloud.

---

## 🔮 Future Improvements

Potential future enhancements include:

* [ ] Real-time MySQL integration with Streamlit
* [ ] User authentication
* [ ] Explainable AI using SHAP
* [ ] Feature importance visualization
* [ ] Advanced risk scoring
* [ ] Loan portfolio analytics
* [ ] Automated model retraining
* [ ] Model monitoring
* [ ] Data drift detection
* [ ] Docker deployment
* [ ] CI/CD pipeline

---

## 📸 Screenshots

Add screenshots of the deployed application here.

Example:

```text
docs/
├── dashboard.png
├── prediction.png
├── history.png
└── analytics.png
```

Then add them to this section:

```markdown
![Credit Risk Dashboard](docs/dashboard.png)

![Credit Risk Prediction](docs/prediction.png)
```

---

## 👨‍💻 Developer

**URADI NARSIMULU**

Computer Science & Engineering Student
Interested in:

* Software Development
* Data Analytics
* Machine Learning
* Artificial Intelligence
* Generative AI

### 🔗 Links

* **Live Demo:** https://credit-risk-platform1.streamlit.app/
* **GitHub:** https://github.com/uradinarsimulu45/credit-risk-platform

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
