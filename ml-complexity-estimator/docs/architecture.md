# System Architecture

## Overview
The ML Model Complexity Estimator consists of two main components:
1.  **Web Extension (Chrome/Edge)**: Captures code from GitHub and displays results.
2.  **FastAPI Backend**: Analyzes the Python code to estimate complexity.

## Data Flow
1.  User clicks "Analyze" on a GitHub file page.
2.  `contentScript.js` scrapes the code from the DOM.
3.  `popup.js` sends the code to the backend (`POST /api/analyze`).
4.  Backend parses AST, detects imports/models, and maps them to complexity rules.
5.  Backend returns JSON response.
6.  `popup.js` renders the complexity card.

## Backend Modules
- **ASTParser**: Uses Python's `ast` module for safe parsing.
- **MLDetector**: Identifies supported frameworks (sklearn, torch, tf).
- **ComplexityEstimator**: Applies heuristic rules to detected models.
