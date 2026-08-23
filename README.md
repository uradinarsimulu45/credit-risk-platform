\# Credit Risk Platform



AI-powered credit risk assessment platform that predicts the probability of loan default using a machine learning model.



\## Overview



The Credit Risk Platform evaluates loan applications using a trained Random Forest classification model.



The system provides:



\* Loan default risk prediction

\* Default probability estimation

\* Low-risk and high-risk classification

\* Prediction history

\* Model statistics

\* MySQL database integration

\* REST API using FastAPI

\* Professional web-based frontend



\## Architecture



```text

&#x20;                   Credit Risk Platform

&#x20;                           |

&#x20;            +--------------+--------------+

&#x20;            |                             |

&#x20;       Frontend                       FastAPI

&#x20;      HTML/CSS/JS                       |

&#x20;            |                    +-------+-------+

&#x20;            |                    |               |

&#x20;            |                ML Model         MySQL

&#x20;            |             Random Forest          |

&#x20;            |                    |               |

&#x20;            +--------------------+---------------+

&#x20;                                 |

&#x20;                        Prediction History

&#x20;                          \& Statistics

```



\## Technology Stack



\### Frontend



\* HTML5

\* CSS3

\* JavaScript



\### Backend



\* Python

\* FastAPI

\* Uvicorn



\### Machine Learning



\* Scikit-learn

\* Random Forest

\* Joblib

\* Pandas



\### Database



\* MySQL

\* SQLAlchemy

\* PyMySQL



\## Project Structure



```text

credit-risk-platform/

│

├── backend/

│   └── app/

│       ├── main.py

│       └── database.py

│

├── frontend/

│   ├── index.html

│   ├── style.css

│   └── script.js

│

├── ml/

│   ├── models/

│   │   └── best\_credit\_risk\_model.joblib

│   └── ...

│

├── data/

│   ├── raw/

│   └── processed/

│

├── tests/

│

├── requirements.txt

├── README.md

└── .gitignore

```



\## Features



\### 1. Loan Application



Users can enter:



\* Loan amount

\* Loan term

\* Interest rate

\* Monthly installment

\* Grade

\* Sub-grade

\* Employment length

\* Home ownership

\* Annual income

\* Verification status

\* Loan purpose

\* Debt-to-income ratio

\* Delinquencies

\* Open accounts

\* Public records

\* Revolving balance

\* Revolving utilization

\* Total accounts

\* Application type



\### 2. Credit Risk Prediction



The Random Forest model generates:



\* Prediction class

\* Risk classification

\* Estimated default probability

\* Prediction ID



Example:



```json

{

&#x20; "prediction": 1,

&#x20; "risk": "High Risk",

&#x20; "default\_probability": 0.55,

&#x20; "prediction\_id": 10

}

```



\### 3. MySQL Prediction History



Every prediction is stored in the `prediction\_history` table.



The table contains:



```text

id

timestamp

prediction

risk

default\_probability

```



\### 4. Model Statistics



The dashboard displays:



\* Total predictions

\* Low-risk predictions

\* High-risk predictions

\* Average default probability

\* High-risk percentage



\## Database Setup



Create the database in MySQL:



```sql

CREATE DATABASE credit\_risk;

```



Select the database:



```sql

USE credit\_risk;

```



The application uses the `prediction\_history` table for storing prediction results.



\## Environment Variables



Create a `.env` file in the project root:



```text

DB\_USER=root

DB\_PASSWORD=your\_mysql\_password

DB\_HOST=localhost

DB\_PORT=3306

DB\_NAME=credit\_risk

```



Do not commit `.env` to GitHub.



\## Installation



Clone the repository:



```bash

git clone https://github.com/uradinarsimulu45/credit-risk-platform.git

cd credit-risk-platform

```



Create a virtual environment:



```bash

python -m venv venv

```



Activate it on Windows:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Run the Backend



```bash

python -m uvicorn backend.app.main:app --reload

```



Backend:



```text

http://127.0.0.1:8000

```



Swagger API documentation:



```text

http://127.0.0.1:8000/docs

```



\## Run the Frontend



Open another terminal:



```bash

python -m http.server 5500 --directory frontend

```



Frontend:



```text

http://127.0.0.1:5500

```



\## Database Verification



To view prediction history:



```sql

USE credit\_risk;



SELECT \*

FROM prediction\_history

ORDER BY id DESC;

```



\## Current Development Verification



The integrated system has been tested with:



\* FastAPI backend

\* Random Forest prediction

\* MySQL storage

\* Prediction history

\* Prediction statistics

\* Professional frontend



Example verified statistics during development:



```text

Total Predictions:       10

Low Risk:                 7

High Risk:                3

Average Default Risk:    39.04%

High Risk Rate:          30.00%

```



\## API



The backend provides endpoints for:



\* Health/status

\* Credit risk prediction

\* Prediction history

\* Prediction statistics

\* Clearing prediction history



Full endpoint details are available through FastAPI Swagger:



```text

http://127.0.0.1:8000/docs

```



\## Security Notes



\* Database credentials should be stored in `.env`.

\* `.env` should not be committed to Git.

\* Production deployments should use restricted CORS origins.

\* Production database credentials should use a dedicated database user rather than the MySQL root account.



\## Future Improvements



Possible future enhancements include:



\* User authentication

\* Explainable AI

\* SHAP-based feature importance

\* Model monitoring

\* Advanced analytics dashboard

\* Automated model retraining

\* Cloud deployment

\* Role-based access control



\## Author



URADI NARSIMULU



GitHub:



https://github.com/uradinarsimulu45



\## License



This project is developed for educational and internship project purposes.



