function isGitHub() {
    return window.location.hostname.includes("github.com");
}

function isColab() {
    return window.location.hostname.includes("colab.research.google.com");
}

function isKaggle() {
    return window.location.hostname.includes("kaggle.com");
}

function isJupyter() {
    // Check for common Jupyter indicators in page or URL
    const jupyterIndicators = [
        document.querySelector('.jp-Notebook'),
        document.querySelector('#notebook'),
        document.querySelector('.jupyter-widgets'),
        window.location.pathname.includes('/notebooks/'),
        window.location.pathname.includes('/lab'),
        document.title.toLowerCase().includes('jupyter')
    ];
    return jupyterIndicators.some(indicator => indicator);
}

function getGitHubFilename() {
    // Try to extract filename from various GitHub UI elements
    const selectors = [
        '[data-testid="breadcrumbs-filename"]',
        '.final-path',
        '.js-path-segment:last-child',
        '[data-pjax="#repo-content-pjax-container"] .final-path',
        '.Box-title strong'
    ];
    
    for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (el && el.textContent.trim()) {
            return el.textContent.trim();
        }
    }
    
    // Fallback: extract from URL path
    const pathParts = window.location.pathname.split('/');
    const filename = pathParts[pathParts.length - 1];
    if (filename && (filename.endsWith('.py') || filename.endsWith('.ipynb'))) {
        return filename;
    }
    
    return 'github_file.py';
}

/* ---------- JUPYTER CODE EXTRACTION ---------- */
function extractJupyterCode() {
    // Try multiple selectors for Jupyter Lab and Notebook
    const codeCells = document.querySelectorAll(".jp-CodeCell .jp-Editor, .CodeMirror-code, .input_area pre");
    if (codeCells.length === 0) {
        // Fallback for simple pre tags if structured cells aren't found
        const preTags = document.querySelectorAll("pre");
        if (preTags.length > 0) return Array.from(preTags).map(el => el.innerText).join("\n\n");
    }

    let code = "";
    codeCells.forEach(cell => {
        code += cell.innerText + "\n\n";
    });
    return code.trim();
}

/* ---------- GITHUB CODE EXTRACTION ---------- */
function extractGitHubCode() {
    // Check if viewing a .ipynb file (Jupyter notebook on GitHub)
    if (window.location.pathname.endsWith('.ipynb')) {
        return extractGitHubNotebook();
    }
    
    // Selectors for different GitHub file views (updated for 2024+ UI)
    const selectors = [
        // New React-based code view (2024+)
        '[data-hpc="true"]',
        '.react-code-lines',
        '.react-blob-view-header-sticky + div',
        // Classic blob view
        "table.highlight",
        ".blob-wrapper table",
        ".blob-wrapper",
        // Other views
        ".js-file-line-container",
        ".react-code-view",
        "textarea[aria-label='File content']"
    ];

    for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element) {
            const text = element.innerText.trim();
            if (text.length > 10) {  // Must have actual content
                console.log(`ML Complexity: Found code using selector: ${selector}`);
                return text;
            }
        }
    }
    
    // Fallback: search for code lines directly
    const lineSelectors = [
        ".react-file-line",
        ".blob-code-inner",
        ".js-file-line",
        "[data-key] .react-code-text"
    ];
    
    for (const selector of lineSelectors) {
        const lines = document.querySelectorAll(selector);
        if (lines.length > 0) {
            console.log(`ML Complexity: Found ${lines.length} lines using: ${selector}`);
            return Array.from(lines).map(line => line.innerText).join("\n");
        }
    }

    return "";
}

/* ---------- GITHUB NOTEBOOK EXTRACTION ---------- */
function extractGitHubNotebook() {
    // GitHub renders .ipynb files - extract code cells
    const codeCells = document.querySelectorAll('.blob-code-inner, .highlight pre, .js-file-line');
    if (codeCells.length > 0) {
        return Array.from(codeCells).map(cell => cell.innerText).join("\n");
    }
    
    // Try rendered notebook view
    const notebookCells = document.querySelectorAll('.jp-InputArea-editor, .input_area');
    if (notebookCells.length > 0) {
        return Array.from(notebookCells).map(cell => cell.innerText).join("\n\n");
    }
    
    return "";
}

/* ---------- GOOGLE COLAB EXTRACTION ---------- */
function extractColabCode() {
    // Colab code cells
    const selectors = [
        '.monaco-editor .view-lines',
        '.code-cell .inputarea',
        '.cell-code .monaco-editor',
        '.codecell'
    ];
    
    for (const selector of selectors) {
        const cells = document.querySelectorAll(selector);
        if (cells.length > 0) {
            return Array.from(cells).map(cell => cell.innerText).join("\n\n");
        }
    }
    
    // Fallback - all code mirror instances
    const editors = document.querySelectorAll('.CodeMirror-code, .monaco-mouse-cursor-text');
    if (editors.length > 0) {
        return Array.from(editors).map(e => e.innerText).join("\n\n");
    }
    
    return "";
}

/* ---------- KAGGLE EXTRACTION ---------- */
function extractKaggleCode() {
    // Kaggle notebook cells
    const selectors = [
        '.cell-code .monaco-editor',
        '.render-container pre',
        '.cell__content pre',
        '.CodeMirror-code'
    ];
    
    for (const selector of selectors) {
        const cells = document.querySelectorAll(selector);
        if (cells.length > 0) {
            return Array.from(cells).map(cell => cell.innerText).join("\n\n");
        }
    }
    
    return "";
}

/* ---------- MESSAGE HANDLER ---------- */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action !== "getCode") return;

    try {
        if (isGitHub()) {
            const code = extractGitHubCode();
            if (!code) {
                sendResponse({ error: "No code found on this GitHub page. Make sure you're viewing a Python file." });
                return;
            }

            sendResponse({
                code,
                filename: getGitHubFilename()
            });
        }

        else if (isColab()) {
            const code = extractColabCode();
            if (!code) {
                sendResponse({ error: "No code cells found in Colab notebook." });
                return;
            }

            sendResponse({
                code,
                filename: "colab_notebook.py"
            });
        }

        else if (isKaggle()) {
            const code = extractKaggleCode();
            if (!code) {
                sendResponse({ error: "No code cells found in Kaggle notebook." });
                return;
            }

            sendResponse({
                code,
                filename: "kaggle_notebook.py"
            });
        }

        else if (isJupyter()) {
            const code = extractJupyterCode();
            if (!code) {
                sendResponse({ error: "No code cells found in notebook." });
                return;
            }

            sendResponse({
                code,
                filename: "notebook.py"
            });
        }

        else {
            sendResponse({ error: "This page is not supported. Open a Python file on GitHub, Colab, Kaggle, or Jupyter." });
        }

    } catch (err) {
        sendResponse({ error: "Extraction failed: " + err.message });
    }

    return true;
});
