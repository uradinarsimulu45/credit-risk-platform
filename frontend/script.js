// ============================================================
// CREDIT RISK PLATFORM - FRONTEND JAVASCRIPT
// Day 14
// ============================================================


// ============================================================
// API CONFIGURATION
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";


// ============================================================
// DOM ELEMENTS
// ============================================================

const form = document.getElementById("predictionForm");

const result = document.getElementById("result");

const predictButton =
    document.getElementById("predictButton");

const resetButton =
    document.getElementById("resetButton");


// ============================================================
// HELPER FUNCTIONS
// ============================================================

function getNumber(id) {

    return Number(
        document.getElementById(id).value
    );

}


function getValue(id) {

    return document.getElementById(id).value;

}


// ============================================================
// BUILD PREDICTION REQUEST
// ============================================================

function buildRequestData() {

    return {

        loan_amnt:
            getNumber("loan_amnt"),

        term:
            getValue("term"),

        int_rate:
            getNumber("int_rate"),

        installment:
            getNumber("installment"),

        grade:
            getValue("grade"),

        sub_grade:
            getValue("sub_grade"),

        emp_length:
            getValue("emp_length"),

        home_ownership:
            getValue("home_ownership"),

        annual_inc:
            getNumber("annual_inc"),

        verification_status:
            getValue("verification_status"),

        purpose:
            getValue("purpose"),

        dti:
            getNumber("dti"),

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
// SHOW LOADING
// ============================================================

function showLoading() {

    result.innerHTML = `

        <div class="loading">

            <h3>
                Analyzing Application...
            </h3>

            <p>
                Please wait while the AI model
                evaluates the loan.
            </p>

        </div>

    `;

}


// ============================================================
// SHOW PREDICTION RESULT
// ============================================================

function showResult(data) {

    const probability =
        (
            Number(data.default_probability) * 100
        ).toFixed(2);


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

                <strong>
                    ${data.prediction}
                </strong>

            </div>

        </div>

    `;

}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(message) {

    result.innerHTML = `

        <div class="error-message">

            <strong>
                Prediction Error
            </strong>

            <p>
                ${message}
            </p>

        </div>

    `;

}


// ============================================================
// PREDICTION FORM
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


            const data =
                buildRequestData();


            try {

                const response =
                    await fetch(
                        `${API_BASE_URL}/predict`,
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


                if (!response.ok) {

                    throw new Error(
                        responseData.detail ||
                        "Prediction request failed."
                    );

                }


                // Show prediction

                showResult(responseData);


                // Refresh history

                loadPredictionHistory();


                // Refresh statistics

                loadStatistics();

            }


            catch (error) {

                console.error(
                    "Prediction error:",
                    error
                );


                showError(
                    error.message ||
                    "Failed to fetch prediction."
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
// RESET FORM
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

                    <h3>
                        No Prediction Yet
                    </h3>

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
// PREDICTION HISTORY
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


    if (!historyBody) {

        return;

    }


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/history`
            );


        if (!response.ok) {

            throw new Error(
                "Failed to load history"
            );

        }


        const data =
            await response.json();


        historyBody.innerHTML = "";


        // No predictions

        if (
            !data.history ||
            data.history.length === 0
        ) {

            if (historyMessage) {

                historyMessage.textContent =
                    "No predictions available.";

            }

            return;

        }


        if (historyMessage) {

            historyMessage.textContent =
                `${data.count} prediction(s) recorded.`;

        }


        // Display newest first

        data.history
            .slice()
            .reverse()
            .forEach(
                record => {

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


                    const riskClass =
                        record.risk === "High Risk"
                            ? "high-risk"
                            : "low-risk";


                    row.innerHTML = `

                        <td>
                            ${record.timestamp}
                        </td>


                        <td>

                            <span
                                class="risk-badge ${riskClass}"
                            >
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


                    historyBody.appendChild(
                        row
                    );

                }
            );

    }


    catch (error) {

        console.error(
            "History error:",
            error
        );


        if (historyMessage) {

            historyMessage.textContent =
                "Failed to load prediction history.";

        }

    }

}


// ============================================================
// REFRESH HISTORY BUTTON
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
// CLEAR HISTORY
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
                        `${API_BASE_URL}/history`,
                        {
                            method: "DELETE"
                        }
                    );


                if (!response.ok) {

                    throw new Error(
                        "Failed to clear history"
                    );

                }


                await loadPredictionHistory();


                await loadStatistics();


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
                    "Failed to clear prediction history."
                );

            }

        }
    );

}


// ============================================================
// MODEL STATISTICS
// ============================================================

async function loadStatistics() {

    const totalPredictions =
        document.getElementById(
            "totalPredictions"
        );


    const lowRiskCount =
        document.getElementById(
            "lowRiskCount"
        );


    const highRiskCount =
        document.getElementById(
            "highRiskCount"
        );


    const averageRisk =
        document.getElementById(
            "averageRisk"
        );


    const highRiskRate =
        document.getElementById(
            "highRiskRate"
        );


    const statsMessage =
        document.getElementById(
            "statsMessage"
        );


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/stats`
            );


        if (!response.ok) {

            throw new Error(
                "Failed to load statistics"
            );

        }


        const data =
            await response.json();


        // Total predictions

        if (totalPredictions) {

            totalPredictions.textContent =
                data.total_predictions;

        }


        // Low risk

        if (lowRiskCount) {

            lowRiskCount.textContent =
                data.low_risk;

        }


        // High risk

        if (highRiskCount) {

            highRiskCount.textContent =
                data.high_risk;

        }


        // Average default probability

        if (averageRisk) {

            averageRisk.textContent =
                (
                    Number(
                        data.average_default_probability
                    ) * 100
                ).toFixed(2) + "%";

        }


        // High risk percentage

        if (highRiskRate) {

            highRiskRate.textContent =
                Number(
                    data.high_risk_percentage
                ).toFixed(2) + "%";

        }


        if (statsMessage) {

            statsMessage.textContent =
                "Statistics updated successfully.";

        }

    }


    catch (error) {

        console.error(
            "Statistics error:",
            error
        );


        if (statsMessage) {

            statsMessage.textContent =
                "Failed to load statistics.";

        }

    }

}


// ============================================================
// REFRESH STATISTICS BUTTON
// ============================================================

const refreshStats =
    document.getElementById(
        "refreshStats"
    );


if (refreshStats) {

    refreshStats.addEventListener(
        "click",
        loadStatistics
    );

}


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        loadPredictionHistory();

        loadStatistics();

    }
);