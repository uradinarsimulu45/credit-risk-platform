// ============================================================
// Credit Risk Platform - Frontend JavaScript
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";
const PREDICT_URL = `${API_BASE_URL}/predict`;
const HISTORY_URL = `${API_BASE_URL}/history`;


// ============================================================
// DOM Elements
// ============================================================

const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");


// ============================================================
// Helper Functions
// ============================================================

function getNumber(id) {
    const element = document.getElementById(id);

    if (!element) {
        throw new Error(`Field not found: ${id}`);
    }

    return Number(element.value);
}


function getValue(id) {
    const element = document.getElementById(id);

    if (!element) {
        throw new Error(`Field not found: ${id}`);
    }

    return element.value;
}


// ============================================================
// Build API Request
// ============================================================

function buildRequestData() {

    return {

        loan_amnt: getNumber("loan_amnt"),

        term: getValue("term"),

        int_rate: getNumber("int_rate"),

        installment: getNumber("installment"),

        grade: getValue("grade"),

        sub_grade: getValue("sub_grade"),

        emp_length: getValue("emp_length"),

        home_ownership: getValue("home_ownership"),

        annual_inc: getNumber("annual_inc"),

        verification_status:
            getValue("verification_status"),

        purpose: getValue("purpose"),

        dti: getNumber("dti"),

        delinq_2yrs:
            getNumber("delinq_2yrs"),

        open_acc:
            getNumber("open_acc"),

        pub_rec:
            getNumber("pub_rec"),

        revol_bal:
            getNumber("revol_bal"),

        revol_util:
            getNumber("revol_util"),

        total_acc:
            getNumber("total_acc"),

        application_type:
            getValue("application_type")
    };
}


// ============================================================
// Loading State
// ============================================================

function showLoading() {

    result.innerHTML = `
        <div class="loading">

            <h3>Analyzing Application...</h3>

            <p>
                Please wait while the AI model
                evaluates the loan.
            </p>

        </div>
    `;
}


// ============================================================
// Display Prediction Result
// ============================================================

function showResult(data) {

    const probability =
        (Number(data.default_probability) * 100)
        .toFixed(2);

    const riskClass =
        data.risk === "High Risk"
            ? "high-risk"
            : "low-risk";

    result.innerHTML = `

        <div class="risk-result">

            <div class="risk-badge ${riskClass}">
                ${data.risk}
            </div>

            <div class="probability">
                ${probability}%
            </div>

            <div class="probability-label">
                Estimated Default Probability
            </div>

            <div class="prediction-value">

                Model Prediction:
                <strong>${data.prediction}</strong>

            </div>

        </div>

    `;

    // Refresh history after successful prediction
    loadPredictionHistory();
}


// ============================================================
// Display Error
// ============================================================

function showError(message) {

    result.innerHTML = `

        <div class="error-message">

            <strong>Prediction Error</strong>

            <p>${message}</p>

        </div>

    `;
}


// ============================================================
// Prediction Form
// ============================================================

if (form) {

    form.addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();

            predictButton.disabled = true;

            predictButton.textContent =
                "Analyzing...";

            showLoading();

            try {

                const data =
                    buildRequestData();

                console.log(
                    "Sending prediction request:",
                    data
                );

                const response =
                    await fetch(
                        PREDICT_URL,
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify(data)
                        }
                    );

                const responseData =
                    await response.json();

                console.log(
                    "Prediction response:",
                    responseData
                );


                if (!response.ok) {

                    throw new Error(
                        responseData.detail ||
                        "Prediction request failed."
                    );
                }


                showResult(responseData);

            }

            catch (error) {

                console.error(
                    "Prediction error:",
                    error
                );

                showError(
                    error.message
                );

            }

            finally {

                predictButton.disabled =
                    false;

                predictButton.textContent =
                    "Predict Credit Risk";
            }
        }
    );
}


// ============================================================
// Reset Form
// ============================================================

if (resetButton) {

    resetButton.addEventListener(
        "click",
        function() {

            form.reset();

            result.innerHTML = `

                <div class="result-placeholder">

                    <div class="placeholder-icon">
                        ?
                    </div>

                    <h3>No Prediction Yet</h3>

                    <p>
                        Submit the loan application
                        to receive an AI-powered
                        credit risk assessment.
                    </p>

                </div>

            `;
        }
    );
}


// ============================================================
// Load Prediction History
// ============================================================

async function loadPredictionHistory() {

    const historyMessage =
        document.getElementById(
            "historyMessage"
        );

    const historyBody =
        document.getElementById(
            "historyBody"
        );


    if (!historyMessage || !historyBody) {
        return;
    }


    try {

        const response =
            await fetch(HISTORY_URL);


        if (!response.ok) {

            throw new Error(
                "Failed to load prediction history."
            );
        }


        const data =
            await response.json();


        historyBody.innerHTML = "";


        if (
            !data.history ||
            data.history.length === 0
        ) {

            historyMessage.textContent =
                "No predictions available.";

            return;
        }


        historyMessage.textContent =
            `${data.count} prediction(s) recorded.`;


        data.history
            .slice()
            .reverse()
            .forEach(
                function(record) {

                    const row =
                        document.createElement(
                            "tr"
                        );


                    const probability =
                        (
                            Number(
                                record.default_probability
                            ) * 100
                        ).toFixed(2);


                    row.innerHTML = `

                        <td>
                            ${record.timestamp}
                        </td>

                        <td>
                            <span class="risk-badge">
                                ${record.risk}
                            </span>
                        </td>

                        <td>
                            ${probability}%
                        </td>

                        <td>
                            ${record.prediction}
                        </td>

                    `;


                    historyBody.appendChild(row);
                }
            );

    }

    catch (error) {

        console.error(
            "History error:",
            error
        );

        historyMessage.textContent =
            "Failed to load prediction history.";
    }
}


// ============================================================
// Refresh History Button
// ============================================================

const refreshHistory =
    document.getElementById(
        "refreshHistory"
    );


if (refreshHistory) {

    refreshHistory.addEventListener(
        "click",
        loadPredictionHistory
    );
}


// ============================================================
// Clear History
// ============================================================

const clearHistory =
    document.getElementById(
        "clearHistory"
    );


if (clearHistory) {

    clearHistory.addEventListener(
        "click",
        async function() {

            const confirmed =
                confirm(
                    "Are you sure you want to clear prediction history?"
                );


            if (!confirmed) {
                return;
            }


            try {

                const response =
                    await fetch(
                        HISTORY_URL,
                        {
                            method: "DELETE"
                        }
                    );


                if (!response.ok) {

                    const errorData =
                        await response.json()
                            .catch(
                                () => ({})
                            );

                    throw new Error(
                        errorData.detail ||
                        "Failed to clear history."
                    );
                }


                await loadPredictionHistory();


                alert(
                    "Prediction history cleared successfully."
                );

            }

            catch (error) {

                console.error(
                    "Clear history error:",
                    error
                );

                alert(
                    error.message
                );
            }
        }
    );
}


// ============================================================
// Load History When Page Opens
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        loadPredictionHistory();

    }
);