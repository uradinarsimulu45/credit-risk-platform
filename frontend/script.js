// =========================================================
// CREDITRISK AI — FRONTEND APPLICATION
// Connects frontend with FastAPI + MySQL
// =========================================================

const API_BASE_URL = "http://127.0.0.1:8000";


// =========================================================
// DOM ELEMENTS
// =========================================================

const predictionForm = document.getElementById("predictionForm");
const resetButton = document.getElementById("resetButton");
const predictButton = document.getElementById("predictButton");

const resultContainer = document.getElementById("result");

const totalPredictions = document.getElementById("totalPredictions");
const lowRisk = document.getElementById("lowRisk");
const highRisk = document.getElementById("highRisk");
const averageRisk = document.getElementById("averageRisk");
const highRiskRate = document.getElementById("highRiskRate");

const refreshStatsButton = document.getElementById("refreshStats");
const statsMessage = document.getElementById("statsMessage");

const refreshHistoryButton = document.getElementById("refreshHistory");
const clearHistoryButton = document.getElementById("clearHistory");

const historyBody = document.getElementById("historyBody");
const historyMessage = document.getElementById("historyMessage");


// =========================================================
// API HELPER
// =========================================================

async function apiRequest(endpoint, options = {}) {

    const response = await fetch(
        `${API_BASE_URL}${endpoint}`,
        {
            ...options,

            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            }
        }
    );


    let data;

    try {

        data = await response.json();

    } catch (error) {

        throw new Error(
            `Server returned an invalid response (${response.status})`
        );

    }


    if (!response.ok) {

        const message =
            data.detail ||
            data.message ||
            `API request failed with status ${response.status}`;

        throw new Error(message);

    }


    return data;
}


// =========================================================
// FORMAT NUMBER
// =========================================================

function formatPercentage(value) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "0.00%";
    }

    return `${(number * 100).toFixed(2)}%`;
}


// =========================================================
// FORMAT TIMESTAMP
// =========================================================

function formatTimestamp(timestamp) {

    if (!timestamp) {
        return "—";
    }


    try {

        const date = new Date(timestamp);

        if (Number.isNaN(date.getTime())) {
            return timestamp;
        }


        return date.toLocaleString(
            "en-IN",
            {
                dateStyle: "medium",
                timeStyle: "short"
            }
        );

    } catch (error) {

        return timestamp;

    }
}


// =========================================================
// LOADING RESULT
// =========================================================

function showResultLoading() {

    resultContainer.innerHTML = `

        <div class="loading">

            <div class="spinner"></div>

            <span>
                Analyzing credit risk...
            </span>

        </div>

    `;
}


// =========================================================
// DISPLAY ERROR
// =========================================================

function showResultError(message) {

    resultContainer.innerHTML = `

        <div class="error-message">

            <strong>Prediction Error</strong>

            <br>

            <span>${escapeHTML(message)}</span>

        </div>

    `;
}


// =========================================================
// ESCAPE HTML
// =========================================================

function escapeHTML(value) {

    const div = document.createElement("div");

    div.textContent = value ?? "";

    return div.innerHTML;
}


// =========================================================
// DISPLAY PREDICTION
// =========================================================

function displayPrediction(data) {

    const prediction = Number(data.prediction);

    const probability = Number(data.default_probability);

    const risk =
        data.risk ||
        (prediction === 1 ? "High Risk" : "Low Risk");


    const isHighRisk =
        prediction === 1 ||
        risk.toLowerCase().includes("high");


    const riskClass =
        isHighRisk ? "high" : "low";


    const icon =
        isHighRisk ? "!" : "✓";


    resultContainer.innerHTML = `

        <div class="risk-result ${riskClass}">

            <div class="risk-badge ${riskClass}">

                <span>${icon}</span>

                <span>
                    ${escapeHTML(risk)}
                </span>

            </div>


            <div class="risk-percentage">

                ${formatPercentage(probability)}

            </div>


            <div class="risk-caption">

                Estimated Default Probability

            </div>


            <div class="risk-divider"></div>


            <div class="prediction-value">

                Model Prediction:

                <strong>
                    ${prediction}
                </strong>

            </div>


            ${
                data.prediction_id !== undefined
                    ? `
                        <div class="prediction-id">

                            Prediction ID:
                            #${escapeHTML(data.prediction_id)}

                        </div>
                      `
                    : ""
            }

        </div>

    `;
}


// =========================================================
// COLLECT FORM DATA
// =========================================================

function getFormData() {

    return {

        loan_amnt:
            Number(
                document.getElementById("loan_amnt").value
            ),

        term:
            document.getElementById("term").value,

        int_rate:
            Number(
                document.getElementById("int_rate").value
            ),

        installment:
            Number(
                document.getElementById("installment").value
            ),

        grade:
            document.getElementById("grade").value,

        sub_grade:
            document.getElementById("sub_grade").value,

        emp_length:
            document.getElementById("emp_length").value,

        home_ownership:
            document.getElementById("home_ownership").value,

        annual_inc:
            Number(
                document.getElementById("annual_inc").value
            ),

        verification_status:
            document.getElementById("verification_status").value,

        purpose:
            document.getElementById("purpose").value,

        dti:
            Number(
                document.getElementById("dti").value
            ),

        delinq_2yrs:
            Number(
                document.getElementById("delinq_2yrs").value
            ),

        open_acc:
            Number(
                document.getElementById("open_acc").value
            ),

        pub_rec:
            Number(
                document.getElementById("pub_rec").value
            ),

        revol_bal:
            Number(
                document.getElementById("revol_bal").value
            ),

        revol_util:
            Number(
                document.getElementById("revol_util").value
            ),

        total_acc:
            Number(
                document.getElementById("total_acc").value
            ),

        application_type:
            document.getElementById("application_type").value
    };
}


// =========================================================
// VALIDATE FORM
// =========================================================

function validateForm(data) {

    for (const [key, value] of Object.entries(data)) {

        if (
            value === "" ||
            value === null ||
            value === undefined
        ) {

            return `Please provide ${key.replaceAll("_", " ")}.`;

        }


        if (
            typeof value === "number" &&
            !Number.isFinite(value)
        ) {

            return `Please enter a valid value for ${key.replaceAll("_", " ")}.`;

        }

    }


    return null;
}


// =========================================================
// PREDICTION
// =========================================================

async function predictCreditRisk(event) {

    event.preventDefault();


    const formData = getFormData();


    const validationError =
        validateForm(formData);


    if (validationError) {

        showResultError(validationError);

        return;

    }


    predictButton.disabled = true;

    predictButton.innerHTML = `
        <span>Analyzing...</span>
    `;


    showResultLoading();


    try {

        const data =
            await apiRequest(
                "/predict",
                {
                    method: "POST",
                    body: JSON.stringify(formData)
                }
            );


        displayPrediction(data);


        // Refresh MySQL-backed dashboard
        await Promise.all([
            loadStatistics(),
            loadHistory()
        ]);


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        showResultError(
            error.message ||
            "Unable to connect to the prediction API."
        );

    } finally {

        predictButton.disabled = false;

        predictButton.innerHTML = `
            <span>Predict Credit Risk</span>
        `;

    }

}


// =========================================================
// LOAD STATISTICS
// =========================================================

async function loadStatistics() {

    try {

        const data =
            await apiRequest("/stats");


        const total =
            Number(data.total_predictions ?? 0);

        const low =
            Number(data.low_risk ?? 0);

        const high =
            Number(data.high_risk ?? 0);

        const average =
            Number(
                data.average_default_probability ?? 0
            );

        const highRate =
            Number(
                data.high_risk_percentage ?? 0
            );


        totalPredictions.textContent =
            total.toLocaleString();


        lowRisk.textContent =
            low.toLocaleString();


        highRisk.textContent =
            high.toLocaleString();


        /*
         * Your API currently returns average probability
         * as a decimal such as 0.3579.
         */
        averageRisk.textContent =
            formatPercentage(average);


        /*
         * high_risk_percentage may be returned either as
         * 11.11 or 0.1111 depending on backend implementation.
         *
         * Detect the format automatically.
         */
        if (highRate <= 1) {

            highRiskRate.textContent =
                formatPercentage(highRate);

        } else {

            highRiskRate.textContent =
                `${highRate.toFixed(2)}%`;

        }


        if (statsMessage) {

            statsMessage.textContent =
                "Statistics updated successfully.";

        }

    } catch (error) {

        console.error(
            "Statistics error:",
            error
        );


        if (statsMessage) {

            statsMessage.textContent =
                `Unable to load statistics: ${error.message}`;

            statsMessage.style.color =
                "#dc2626";

        }

    }

}


// =========================================================
// LOAD HISTORY
// =========================================================

async function loadHistory() {

    try {

        if (historyMessage) {

            historyMessage.textContent =
                "Loading prediction history...";

        }


        const data =
            await apiRequest("/history");


        /*
         * Support multiple possible backend response formats:
         *
         * 1. [ {...}, {...} ]
         *
         * 2. { "history": [...] }
         *
         * 3. { "predictions": [...] }
         */

        let records;


        if (Array.isArray(data)) {

            records = data;

        } else if (Array.isArray(data.history)) {

            records = data.history;

        } else if (Array.isArray(data.predictions)) {

            records = data.predictions;

        } else {

            records = [];

        }


        renderHistory(records);


    } catch (error) {

        console.error(
            "History error:",
            error
        );


        historyBody.innerHTML = `

            <tr>

                <td
                    colspan="4"
                    class="empty-history"
                >

                    Unable to load prediction history.

                </td>

            </tr>

        `;


        if (historyMessage) {

            historyMessage.textContent =
                `Error: ${error.message}`;

        }

    }

}


// =========================================================
// RENDER HISTORY
// =========================================================

function renderHistory(records) {

    historyBody.innerHTML = "";


    if (!records || records.length === 0) {

        historyBody.innerHTML = `

            <tr>

                <td
                    colspan="4"
                    class="empty-history"
                >

                    No prediction history available.

                </td>

            </tr>

        `;


        if (historyMessage) {

            historyMessage.textContent =
                "0 predictions recorded.";

        }

        return;

    }


    /*
     * Newest records first.
     */
    const sortedRecords =
        [...records].sort(
            (a, b) => {

                const idA =
                    Number(a.id ?? 0);

                const idB =
                    Number(b.id ?? 0);

                return idB - idA;

            }
        );


    sortedRecords.forEach(record => {

        const prediction =
            Number(record.prediction);


        const probability =
            Number(
                record.default_probability ?? 0
            );


        const risk =
            record.risk ||
            (prediction === 1
                ? "High Risk"
                : "Low Risk");


        const isHigh =
            prediction === 1 ||
            risk.toLowerCase().includes("high");


        const row =
            document.createElement("tr");


        const timestamp =
            record.timestamp ||
            record.created_at ||
            record.time;


        row.innerHTML = `

            <td>
                ${escapeHTML(
                    formatTimestamp(timestamp)
                )}
            </td>


            <td>

                <span
                    class="${
                        isHigh
                            ? "history-high"
                            : "history-low"
                    }"
                >

                    ${escapeHTML(risk)}

                </span>

            </td>


            <td>

                ${formatPercentage(probability)}

            </td>


            <td>

                <strong>
                    ${prediction}
                </strong>

            </td>

        `;


        historyBody.appendChild(row);

    });


    if (historyMessage) {

        historyMessage.textContent =
            `${sortedRecords.length} prediction(s) recorded.`;

    }

}


// =========================================================
// RESET FORM
// =========================================================

function resetForm() {

    predictionForm.reset();


    resultContainer.innerHTML = `

        <div class="result-placeholder">

            <div class="placeholder-icon">
                ?
            </div>

            <h3>No Prediction Yet</h3>

            <p>
                Submit the loan application to receive
                an AI-powered credit risk assessment.
            </p>

        </div>

    `;

}


// =========================================================
// CLEAR HISTORY
// =========================================================

async function clearPredictionHistory() {

    const confirmed =
        window.confirm(
            "Are you sure you want to delete all prediction history?"
        );


    if (!confirmed) {
        return;
    }


    try {

        clearHistoryButton.disabled = true;

        clearHistoryButton.textContent =
            "Clearing...";


        await apiRequest(
            "/history",
            {
                method: "DELETE"
            }
        );


        await Promise.all([
            loadStatistics(),
            loadHistory()
        ]);


        if (statsMessage) {

            statsMessage.textContent =
                "Prediction history cleared successfully.";

            statsMessage.style.color =
                "#16a34a";

        }


    } catch (error) {

        console.error(
            "Clear history error:",
            error
        );


        if (statsMessage) {

            statsMessage.textContent =
                `Unable to clear history: ${error.message}`;

            statsMessage.style.color =
                "#dc2626";

        }

    } finally {

        clearHistoryButton.disabled = false;

        clearHistoryButton.textContent =
            "Clear History";

    }

}


// =========================================================
// API HEALTH CHECK
// =========================================================

async function checkAPIHealth() {

    const statusContainer =
        document.querySelector(".api-status");


    if (!statusContainer) {
        return;
    }


    try {

        const data =
            await apiRequest("/");


        if (
            data &&
            (
                data.status === "healthy" ||
                data.message
            )
        ) {

            statusContainer.innerHTML = `

                <span class="status-dot"></span>

                <span>API Online</span>

            `;

        }

    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );


        statusContainer.innerHTML = `

            <span
                class="status-dot"
                style="background:#dc2626; box-shadow:0 0 0 4px rgba(220,38,38,.12);"
            ></span>

            <span style="color:#b91c1c;">
                API Offline
            </span>

        `;

    }

}


// =========================================================
// EVENT LISTENERS
// =========================================================

if (predictionForm) {

    predictionForm.addEventListener(
        "submit",
        predictCreditRisk
    );

}


if (resetButton) {

    resetButton.addEventListener(
        "click",
        resetForm
    );

}


if (refreshStatsButton) {

    refreshStatsButton.addEventListener(
        "click",
        async () => {

            refreshStatsButton.disabled = true;

            refreshStatsButton.textContent =
                "Refreshing...";


            await loadStatistics();


            refreshStatsButton.disabled = false;

            refreshStatsButton.textContent =
                "Refresh Statistics";

        }
    );

}


if (refreshHistoryButton) {

    refreshHistoryButton.addEventListener(
        "click",
        async () => {

            refreshHistoryButton.disabled = true;

            refreshHistoryButton.textContent =
                "Refreshing...";


            await loadHistory();


            refreshHistoryButton.disabled = false;

            refreshHistoryButton.textContent =
                "Refresh History";

        }
    );

}


if (clearHistoryButton) {

    clearHistoryButton.addEventListener(
        "click",
        clearPredictionHistory
    );

}


// =========================================================
// INITIALIZE APPLICATION
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    async () => {

        console.log(
            "CreditRisk AI frontend initialized."
        );


        await checkAPIHealth();


        await Promise.all([
            loadStatistics(),
            loadHistory()
        ]);

    }
);