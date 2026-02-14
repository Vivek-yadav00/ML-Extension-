const analyzeBtn = document.getElementById('analyzeBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');

if (!analyzeBtn || !statusDiv || !resultsDiv) {
    console.error("Popup HTML missing required elements");
} else {

    analyzeBtn.addEventListener('click', async () => {
        statusDiv.textContent = "Extracting code...";
        statusDiv.style.color = "#666";
        resultsDiv.innerHTML = "";
        resultsDiv.style.display = "none";

        let tab;
        try {
            [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        } catch (e) {
            statusDiv.textContent = "Error accessing tab: " + e.message;
            statusDiv.style.color = "red";
            return;
        }

        chrome.tabs.sendMessage(tab.id, { action: "getCode" }, async (response) => {
            if (chrome.runtime.lastError) {
                statusDiv.textContent = "Please reload the page and try again.";
                statusDiv.style.color = "red";
                console.error(chrome.runtime.lastError);
                return;
            }

            if (!response || response.error) {
                statusDiv.textContent = response?.error || "No code found. Open a Python file on GitHub or a Jupyter Notebook.";
                statusDiv.style.color = "red";
                return;
            }

            statusDiv.textContent = "Analyzing " + (response.filename || "code") + "...";

            try {
                // Initialize the local analyzer
                const analyzer = new window.MLAnalyzer();
                const data = analyzer.analyze(response.code, response.filename || "unknown.py");

                displayResults(data);
                statusDiv.textContent = "";

            } catch (err) {
                statusDiv.textContent = "Error: " + err.message;
                statusDiv.style.color = "red";
            }
        });
    });
}

function displayResults(data) {
    resultsDiv.innerHTML = "";
    resultsDiv.style.display = "block";

    // Show detected packages section
    if (data.detected_frameworks && data.detected_frameworks.length > 0) {
        const pkgDiv = document.createElement('div');
        pkgDiv.className = 'frameworks-section';
        pkgDiv.innerHTML = `
            <div class="section-title">Detected Libraries</div>
            <div class="framework-tags">
                ${data.detected_frameworks.map(pkg => `<span class="framework-tag">${pkg}</span>`).join('')}
            </div>
        `;
        resultsDiv.appendChild(pkgDiv);
    }

    // Show models if any
    if (data.models && data.models.length > 0) {
        data.models.forEach(model => {
            const card = document.createElement('div');
            card.className = 'model-card';

            let html = `<div class="model-header">
                <strong>${model.model_type}</strong> 
                <span class="framework-badge">${model.framework}</span>
                <span class="line-number">Line ${model.line_number}</span>
            </div>`;

            // General Description Section
            if (model.description) {
                html += `
                    <div class="section general-info">
                        <div class="section-title">About This Model</div>
                        <p class="summary">${model.description.summary}</p>
                        <p class="use-case"><strong>Use Case:</strong> ${model.description.use_case}</p>
                        ${model.description.pros && model.description.pros.length > 0 ? `
                            <div class="pros-cons">
                                <span class="pros">Pros: ${model.description.pros.slice(0, 2).join(', ')}</span>
                            </div>
                        ` : ''}
                        ${model.description.cons && model.description.cons.length > 0 ? `
                            <div class="pros-cons">
                                <span class="cons">Cons: ${model.description.cons.slice(0, 2).join(', ')}</span>
                            </div>
                        ` : ''}
                    </div>
                `;
            }

            // Complexity Section
            if (model.complexity) {
                html += `
                    <div class="section complexity-info">
                        <div class="section-title">Complexity</div>
                        <div class="complexity-tags">
                            <span class="complexity-tag time">Time: ${model.complexity.time_complexity}</span>
                            <span class="complexity-tag memory">Mem: ${model.complexity.memory_complexity}</span>
                        </div>
                    </div>
                `;
            }

            // File-Specific Analysis Section
            if (model.file_analysis) {
                html += `<div class="section file-analysis">
                    <div class="section-title">Your Code Analysis</div>`;

                if (model.file_analysis.data_info) {
                    html += `<p class="data-info">${model.file_analysis.data_info}</p>`;
                }

                if (model.file_analysis.parameters_used && Object.keys(model.file_analysis.parameters_used).length > 0) {
                    const params = Object.entries(model.file_analysis.parameters_used)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(', ');
                    html += `<p class="params"><strong>Parameters:</strong> ${params}</p>`;
                }

                if (model.file_analysis.suggestions && model.file_analysis.suggestions.length > 0) {
                    html += `<div class="suggestions">
                        <strong>Suggestions:</strong>
                        <ul>
                            ${model.file_analysis.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>`;
                }

                html += `</div>`;
            }

            card.innerHTML = html;
            resultsDiv.appendChild(card);
        });
    } else if (!data.data_operations || data.data_operations.length === 0) {
        resultsDiv.innerHTML += "<p>No ML models or operations detected.</p>";
    }

    // Show data operations if any
    if (data.data_operations && data.data_operations.length > 0) {
        const opsDiv = document.createElement('div');
        opsDiv.className = 'data-ops-section';
        opsDiv.innerHTML = `
            <div class="section-title">Data Operations Found</div>
            <ul>
                ${data.data_operations.map(op => `<li>${op}</li>`).join('')}
            </ul>
        `;
        resultsDiv.appendChild(opsDiv);
    }
}
