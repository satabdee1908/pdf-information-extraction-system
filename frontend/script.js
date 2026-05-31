window.onload = function () {
    document.querySelector(".hero").style.display = "flex";
    document.getElementById("mainApp").style.display = "none";
};

function showApp(event) {
    if (event) event.preventDefault();

    document.querySelector(".hero").style.display = "none";
    document.getElementById("mainApp").style.display = "block";
}

async function uploadFile(event) {
    if (event) event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");

    if (!fileInput.files.length) {
        alert("Please select a file first");
        return;
    }

    loading.innerText = "Processing document... please wait.";
    result.innerHTML = "<p class='placeholder'>Please wait, extracting information...</p>";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:8000/extract-text", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loading.innerText = "Extraction completed.";

        const ai = data.ai_output || {};
        const fields = ai.important_fields || {};
        const entities = ai.important_entities || {};

        result.innerHTML = `
            <div class="card">
                <h2>Document Overview</h2>
                <p><strong>Filename:</strong> ${data.filename || "Not available"}</p>
                <p><strong>Document Type:</strong> ${data.document_type || "Not available"}</p>
                <p><strong>AI Suggested Type:</strong> ${ai.suggested_document_type || "Not available"}</p>
                <p><strong>Extraction Method:</strong> ${data.extraction_method || "Not available"}</p>
            </div>

            <div class="card">
                <h2>AI Summary</h2>
                <p>${ai.summary || "No summary available"}</p>
            </div>

            <div class="card">
                <h2>Key Points</h2>
                <ul>
                    ${(ai.key_points || []).map(point => `<li>${point}</li>`).join("") || "<li>No key points available</li>"}
                </ul>
            </div>

            <div class="card">
                <h2>Important Fields</h2>
                <pre>${JSON.stringify(fields, null, 4)}</pre>
            </div>

            <div class="card">
                <h2>Entities</h2>
                <p><strong>People:</strong> ${(entities.people || []).join(", ") || "None"}</p>
                <p><strong>Organizations:</strong> ${(entities.organizations || []).join(", ") || "None"}</p>
                <p><strong>Dates:</strong> ${(entities.dates || []).join(", ") || "None"}</p>
                <p><strong>Locations:</strong> ${(entities.locations || []).join(", ") || "None"}</p>
            </div>

            <details class="raw-json">
                <summary>View Raw JSON</summary>
                <pre>${JSON.stringify(data, null, 4)}</pre>
            </details>
        `;

    } catch (error) {
        loading.innerText = "Error occurred.";
        result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}