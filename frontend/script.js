const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");


function getNumber(id) {
    return Number(document.getElementById(id).value);
}


function getValue(id) {
    return document.getElementById(id).value;
}


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


function showLoading() {

    result.innerHTML = `
        <div class="loading">
            <h3>Analyzing Application...</h3>
            <p>Please wait while the AI model evaluates the loan.</p>
        </div>
    `;
}


function showResult(data) {

    const probability =
        (data.default_probability * 100).toFixed(2);

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
}


function showError(message) {

    result.innerHTML = `

        <div class="error-message">

            <strong>Prediction Error</strong>

            <p>${message}</p>

        </div>

    `;
}


form.addEventListener("submit", async function(event) {

    event.preventDefault();

    predictButton.disabled = true;

    predictButton.textContent = "Analyzing...";

    showLoading();

    const data = buildRequestData();

    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
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


        showResult(responseData);

    }

    catch (error) {

        console.error(error);

        showError(error.message);

    }

    finally {

        predictButton.disabled = false;

        predictButton.textContent =
            "Predict Credit Risk";

    }

});


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
                    Submit the loan application to receive
                    an AI-powered credit risk assessment.
                </p>

            </div>

        `;

    }
);