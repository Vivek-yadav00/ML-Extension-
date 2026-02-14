/**
 * Local ML Complexity Analyzer for Chrome Extension
 * Ports logic from the Python backend to provide standalone analysis.
 */

class MLAnalyzer {
    constructor() {
        this.frameworks = {
            "sklearn": ["sklearn", "scikit"],
            "tensorflow": ["tensorflow", "keras", "tf"],
            "pytorch": ["torch", "torchvision"],
            "xgboost": ["xgboost"],
            "lightgbm": ["lightgbm"],
            "catboost": ["catboost"],
            "pandas": ["pandas"],
            "numpy": ["numpy"],
            "matplotlib": ["matplotlib"],
            "seaborn": ["seaborn"],
            "scipy": ["scipy"],
            "statsmodels": ["statsmodels"]
        };

        this.knownModels = {
            "LogisticRegression": ["Logistic Regression", "sklearn"],
            "LinearRegression": ["Linear Regression", "sklearn"],
            "Ridge": ["Ridge Regression", "sklearn"],
            "Lasso": ["Lasso Regression", "sklearn"],
            "SVR": ["Support Vector Regressor", "sklearn"],
            "RandomForestClassifier": ["Random Forest Classifier", "sklearn"],
            "RandomForestRegressor": ["Random Forest Regressor", "sklearn"],
            "GradientBoostingClassifier": ["Gradient Boosting Classifier", "sklearn"],
            "GradientBoostingRegressor": ["Gradient Boosting Regressor", "sklearn"],
            "DecisionTreeClassifier": ["Decision Tree Classifier", "sklearn"],
            "DecisionTreeRegressor": ["Decision Tree Regressor", "sklearn"],
            "KNeighborsClassifier": ["K-Nearest Neighbors Classifier", "sklearn"],
            "KNeighborsRegressor": ["K-Nearest Neighbors Regressor", "sklearn"],
            "AdaBoostClassifier": ["AdaBoost Classifier", "sklearn"],
            "AdaBoostRegressor": ["AdaBoost Regressor", "sklearn"],
            "ExtraTreesClassifier": ["Extra Trees Classifier", "sklearn"],
            "ExtraTreesRegressor": ["Extra Trees Regressor", "sklearn"],
            "GaussianNB": ["Gaussian Naive Bayes", "sklearn"],
            "MultinomialNB": ["Multinomial Naive Bayes", "sklearn"],
            "BernoulliNB": ["Bernoulli Naive Bayes", "sklearn"],
            "LinearRegression": ["Linear Regression", "sklearn"],
            "Ridge": ["Ridge Regression", "sklearn"],
            "Lasso": ["Lasso Regression", "sklearn"],
            "ElasticNet": ["Elastic Net", "sklearn"],
            "KMeans": ["K-Means Clustering", "sklearn"],
            "DBSCAN": ["DBSCAN Clustering", "sklearn"],
            "AgglomerativeClustering": ["Agglomerative Clustering", "sklearn"],
            "StandardScaler": ["Standard Scaler", "sklearn"],
            "MinMaxScaler": ["MinMax Scaler", "sklearn"],
            "LabelEncoder": ["Label Encoder", "sklearn"],
            "OneHotEncoder": ["One-Hot Encoder", "sklearn"],
            "PCA": ["Principal Component Analysis", "sklearn"],
            "VotingClassifier": ["Voting Classifier", "sklearn"],
            "BaggingClassifier": ["Bagging Classifier", "sklearn"],
            "StackingClassifier": ["Stacking Classifier", "sklearn"],
            "XGBClassifier": ["XGBoost Classifier", "xgboost"],
            "XGBRegressor": ["XGBoost Regressor", "xgboost"],
            "LGBMClassifier": ["LightGBM Classifier", "lightgbm"],
            "LGBMRegressor": ["LightGBM Regressor", "lightgbm"],
            "CatBoostClassifier": ["CatBoost Classifier", "catboost"],
            "CatBoostRegressor": ["CatBoost Regressor", "catboost"],
            "Linear": ["Linear Layer", "pytorch"],
            "Conv2d": ["Convolutional Layer", "pytorch"],
            "Conv1d": ["1D Convolutional Layer", "pytorch"],
            "LSTM": ["LSTM Layer", "pytorch"],
            "GRU": ["GRU Layer", "pytorch"],
            "Sequential": ["Sequential Model", "pytorch"],
            "Module": ["Neural Network Module", "pytorch"],
            "Dense": ["Dense Layer", "tensorflow"],
            "Conv2D": ["Convolutional Layer", "tensorflow"],
            "Dropout": ["Dropout Layer", "tensorflow"],
            "BatchNormalization": ["Batch Normalization", "tensorflow"],
            // SKLearn Metrics & Model Selection
            "train_test_split": ["Train Test Split", "sklearn"],
            "r2_score": ["R² Score", "sklearn"],
            "mean_absolute_error": ["Mean Absolute Error", "sklearn"],
            "mean_squared_error": ["Mean Squared Error", "sklearn"],
        };

        this.modelInfo = {
            "LogisticRegression": {
                "summary": "A linear classifier that predicts probabilities using the logistic function.",
                "use_case": "Binary/multi-class classification with linearly separable data",
                "pros": ["Fast training and prediction", "Interpretable coefficients", "Works well with small datasets"],
                "cons": ["Assumes linear decision boundary", "May underfit complex patterns"],
                "time": "O(n_samples * n_features * n_classes)",
                "memory": "O(n_features * n_classes)",
            },
            "LinearRegression": {
                "summary": "Fits a linear model to minimize the sum of squared residuals.",
                "use_case": "Predicting continuous values with linear relationships",
                "pros": ["Very fast", "Simple and interpretable", "No hyperparameters to tune"],
                "cons": ["Assumes linear relationship", "Sensitive to outliers"],
                "time": "O(n_samples * n_features²)",
                "memory": "O(n_features)",
            },
            "Ridge": {
                "summary": "Linear regression with L2 regularization to prevent overfitting.",
                "use_case": "Regression when features are correlated",
                "pros": ["Prevents overfitting", "Handles multicollinearity"],
                "cons": ["Doesn't perform feature selection"],
                "time": "O(n_samples * n_features²)",
                "memory": "O(n_features)",
            },
            "Lasso": {
                "summary": "Linear regression with L1 regularization that performs feature selection.",
                "use_case": "Regression with automatic feature selection",
                "pros": ["Performs feature selection", "Creates sparse models"],
                "cons": ["Can be unstable with correlated features"],
                "time": "O(n_samples * n_features²)",
                "memory": "O(n_features)",
            },
            "DecisionTreeClassifier": {
                "summary": "A tree-based classifier that splits data based on feature values.",
                "use_case": "Classification with interpretable decision rules",
                "pros": ["Easy to understand", "Handles non-linear relationships"],
                "cons": ["Prone to overfitting", "Unstable"],
                "time": "O(n_samples * n_features * log(n_samples))",
                "memory": "O(n_nodes)",
            },
            "RandomForestClassifier": {
                "summary": "An ensemble of decision trees trained on random subsets of data.",
                "use_case": "General-purpose classification with good accuracy",
                "pros": ["Reduces overfitting", "Handles high-dimensional data"],
                "cons": ["Less interpretable than single tree", "Memory intensive"],
                "time": "O(n_trees * n_samples * log(n_samples) * n_features)",
                "memory": "O(n_trees * n_leaves)",
            },
            "SVC": {
                "summary": "Support Vector Classifier finds the optimal hyperplane that maximizes margin.",
                "use_case": "Binary classification with clear margin of separation",
                "pros": ["Effective in high dimensions", "Kernel trick for non-linear data"],
                "cons": ["Scales poorly O(n²)", "Sensitive to scaling"],
                "time": "O(n_samples² * n_features) to O(n_samples³)",
                "memory": "O(n_samples * n_features)",
            },
            "KMeans": {
                "summary": "Partitions data into k clusters by minimizing within-cluster variance.",
                "use_case": "Customer segmentation, data compression",
                "pros": ["Simple and fast", "Scales well"],
                "cons": ["Must specify k in advance", "Sensitive to initialization"],
                "time": "O(n_samples * k * n_iterations * n_features)",
                "memory": "O(n_samples * n_features + k * n_features)",
            },
            "StandardScaler": {
                "summary": "Standardizes features by removing mean and scaling to unit variance.",
                "use_case": "Feature scaling before distance-based algorithms",
                "pros": ["Makes features comparable", "Required by many algorithms"],
                "cons": ["Sensitive to outliers"],
                "time": "O(n_samples * n_features)",
                "memory": "O(n_features)",
            },
            "PCA": {
                "summary": "Reduces dimensions by projecting data onto directions of maximum variance.",
                "use_case": "Dimensionality reduction, visualization",
                "pros": ["Reduces overfitting", "Speeds up training"],
                "cons": ["Loses interpretability"],
                "time": "O(n_features² * n_samples + n_features³)",
                "memory": "O(n_features²)",
            },
            "train_test_split": {
                "summary": "Splits arrays or matrices into random train and test subsets.",
                "use_case": "Model validation and evaluation",
                "pros": ["Essential for detecting overfitting", "Randomized for unbiased evaluation"],
                "cons": ["Reduces training data size"],
                "time": "O(n_samples)",
                "memory": "O(n_samples)"
            },
            "r2_score": {
                "summary": "R-squared (coefficient of determination) regression score function.",
                "use_case": "Evaluating regression model performance",
                "pros": ["Standardized measure (0 to 1)", "Interpretability"],
                "cons": ["Can be negative for bad models", "Doesn't indicate error magnitude"],
                "time": "O(n_samples)",
                "memory": "O(1)"
            },
            "mean_absolute_error": {
                "summary": "Mean absolute error regression loss.",
                "use_case": "Evaluating regression accuracy",
                "pros": ["Robust to outliers compared to MSE", "Interpretable units"],
                "cons": [" not differentiable at 0"],
                "time": "O(n_samples)",
                "memory": "O(1)"
            },
            "mean_squared_error": {
                "summary": "Mean squared error regression loss.",
                "use_case": "Evaluating regression accuracy",
                "pros": ["Penalizes large errors heavily", "Differentiable"],
                "cons": ["Sensitive to outliers", "Units are squared"],
                "time": "O(n_samples)",
                "memory": "O(1)"
            }
        };

        this.dataMethods = {
            'read_csv': 'Load CSV file',
            'read_excel': 'Load Excel file',
            'read_json': 'Load JSON file',
            'read_sql': 'Load from SQL database',
            'load': 'Load data',
            'train_test_split': 'Split data into train/test sets',
            'fit': 'Train/fit model',
            'fit_transform': 'Fit and transform data',
            'transform': 'Transform data',
            'predict': 'Make predictions',
            'score': 'Evaluate model',
            'head': 'Preview data',
            'describe': 'Data statistics',
            'r2_score': 'R2 Score Evaluation',
            'mean_absolute_error': 'MAE Evaluation',
            'mean_squared_error': 'MSE Evaluation',
            'plot': 'Create visualization',
            'show': 'Display plot'
        };
    }

    analyze(content, filename = "unknown.py") {
        const lines = content.split('\n');
        const imports = this.extractImports(content);
        const importedNames = this.extractImportedNames(content);
        const frameworks = this.detectFrameworks(imports);
        const dataOps = this.extractDataOperations(content);
        const instantiations = this.extractInstantiations(content);

        const models = [];
        const seen = new Set();

        for (const inst of instantiations) {
            const fullCall = inst.name; // e.g., "np.mean" or "LinearRegression"
            const parts = fullCall.split('.');
            const className = parts.pop();
            const prefix = parts.join('.'); // e.g., "np" or ""

            let modelData = null;

            // 1. Check Exact Match in Known Models
            if (this.knownModels[className]) {
                const [type, framework] = this.knownModels[className];
                modelData = { name: className, type, framework, line: inst.line, params: inst.params };
            }
            // 2. Check via Imports (e.g. from sklearn.linear_model import LinearRegression)
            else if (importedNames[className]) {
                const sourceModule = importedNames[className];
                for (const [framework, keywords] of Object.entries(this.frameworks)) {
                    if (keywords.some(k => sourceModule.toLowerCase().includes(k))) {
                        modelData = {
                            name: className,
                            type: this.formatName(className),
                            framework: framework,
                            line: inst.line,
                            params: inst.params,
                            source: sourceModule // Keep track of source for heuristics
                        };
                        break;
                    }
                }
            }
            // 3. Check via Prefix/Alias (e.g. np.mean, pd.read_csv, metrics.accuracy_score)
            else if (prefix) {
                // Check if prefix maps to a known library import
                // e.g. import numpy as np -> names['np'] = 'numpy'
                const sourceModule = importedNames[prefix] || prefix;

                for (const [framework, keywords] of Object.entries(this.frameworks)) {
                    // Check if the alias source contains framework keywords (e.g. 'numpy' contains 'numpy')
                    if (keywords.some(k => sourceModule.toLowerCase().includes(k))) {
                        modelData = {
                            name: className, // e.g. "mean" or "accuracy_score"
                            type: this.formatName(className),
                            framework: framework,
                            line: inst.line,
                            params: inst.params,
                            source: sourceModule // e.g. "numpy" or "sklearn.metrics"
                        };
                        break;
                    }
                }
            }

            if (modelData) {
                // Filter out non-ML components for clarity (e.g. plotting functions, array creation)
                const ignoredFrameworks = ['matplotlib', 'seaborn', 'numpy', 'pandas', 'scipy', 'statsmodels'];
                const isExplicitML = modelData.source && (
                    modelData.source.includes('sklearn') ||
                    modelData.source.includes('xgboost') ||
                    modelData.source.includes('tensorflow') ||
                    modelData.source.includes('keras') ||
                    modelData.source.includes('torch')
                );

                if (!isExplicitML && ignoredFrameworks.includes(modelData.framework)) {
                    continue;
                }

                const key = `${modelData.name}_${modelData.line}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    // Pass source mainly for heuristics
                    const analysis = this.estimateComplexity(modelData, inst.params, dataOps, modelData.source);
                    models.push({
                        framework: modelData.framework,
                        model_type: modelData.type,
                        model_name: modelData.name,
                        line_number: modelData.line,
                        complexity: analysis.complexity,
                        description: analysis.description,
                        file_analysis: analysis.file_analysis
                    });
                }
            }
        }

        // 4. Add Imported but Unused Models (User Request: "all thing comes")
        // This ensures that if a user imports StandardScaler but hasn't used it yet, it still shows up.
        for (const [name, module] of Object.entries(importedNames)) {
            // Skip if we've already analyzed it as an instantiation
            // We check generic name, as line number doesn't exist for import-only
            // To do this effectively, we need to know if we saw this NAME in instantiations.

            // Check if ANY instantiation matched this name
            const alreadyUsed = models.some(m => m.model_name === name);
            if (alreadyUsed) continue;

            const modelData = {
                name: name,
                type: this.formatName(name),
                framework: 'unknown', // Will detect below
                line: 'Imported',
                params: {},
                source: module
            };

            // Detect framework for this import
            for (const [framework, keywords] of Object.entries(this.frameworks)) {
                if (keywords.some(k => module.toLowerCase().includes(k))) {
                    modelData.framework = framework;
                    break;
                }
            }

            // Filter non-ML imports (noise reduction)
            const ignoredFrameworks = ['matplotlib', 'seaborn', 'numpy', 'pandas', 'scipy', 'statsmodels'];
            const isExplicitML = modelData.source && (
                modelData.source.includes('sklearn') ||
                modelData.source.includes('xgboost') ||
                modelData.source.includes('tensorflow') ||
                modelData.source.includes('keras') ||
                modelData.source.includes('torch')
            );

            // knownModels override
            if (this.knownModels[name]) {
                modelData.type = this.knownModels[name][0];
                modelData.framework = this.knownModels[name][1];
            } else if (!isExplicitML && ignoredFrameworks.includes(modelData.framework)) {
                continue;
            }

            // If we still don't know the framework and it's not a known model, maybe skip?
            // But if it's from 'sklearn.metrics', we want it.
            if (modelData.framework === 'unknown' && !modelData.source.includes('sklearn')) {
                continue;
            }

            const analysis = this.estimateComplexity(modelData, {}, dataOps, modelData.source);

            models.push({
                framework: modelData.framework,
                model_type: modelData.type,
                model_name: modelData.name,
                line_number: 'Imported',
                complexity: analysis.complexity,
                description: analysis.description,
                file_analysis: { ...analysis.file_analysis, estimated_scale: "Imported but not detected in usage" }
            });
        }

        return {
            filename,
            detected_frameworks: frameworks,
            models: models,
            data_operations: dataOps.map(op => `${op.description} (line ${op.line})`)
        };
    }

    extractImports(content) {
        const imports = [];
        // Support both 'import module' and 'from module import ...'
        // Handle indentation and multiple items
        // Fix: Use non-greedy match for import line to avoid swallowing newlines
        const importRegex = /^\s*import\s+([^\n]+)/gm;
        const fromImportRegex = /^\s*from\s+([a-zA-Z0-9_.]+)\s+import/gm;

        let match;
        while ((match = importRegex.exec(content)) !== null) {
            match[1].split(',').forEach(imp => {
                const parts = imp.trim().split(/\s+/);
                if (parts[0]) imports.push(parts[0].split('.')[0]);
            });
        }
        while ((match = fromImportRegex.exec(content)) !== null) {
            if (match[1]) imports.push(match[1].split('.')[0]);
        }
        return [...new Set(imports)];
    }

    extractImportedNames(content) {
        const names = {};
        // from module import name as alias
        const fromImportRegex = /from\s+([a-zA-Z0-9_.]+)\s+import\s+([^#\n]+)/g;
        let match;
        while ((match = fromImportRegex.exec(content)) !== null) {
            const module = match[1];
            const imports = match[2].split(',');
            imports.forEach(imp => {
                const parts = imp.trim().split(/\s+as\s+/);
                const name = parts[parts.length - 1];
                names[name] = module;
            });
        }
        // import module as alias
        const importRegex = /import\s+([a-zA-Z0-9_.]+)(?:\s+as\s+([a-zA-Z0-9_]+))?/g;
        while ((match = importRegex.exec(content)) !== null) {
            const module = match[1];
            const alias = match[2] || module.split('.').pop();
            names[alias] = module;
        }
        return names;
    }

    detectFrameworks(imports) {
        const detected = new Set();
        imports.forEach(imp => {
            const impLower = imp.toLowerCase();
            for (const [framework, keywords] of Object.entries(this.frameworks)) {
                if (keywords.some(k => impLower.includes(k))) {
                    detected.add(framework);
                }
            }
        });
        return [...detected];
    }

    extractDataOperations(content) {
        const ops = [];
        const lines = content.split('\n');
        lines.forEach((line, index) => {
            for (const [method, desc] of Object.entries(this.dataMethods)) {
                // Look for method call: .method( or broad call: method(
                // The broad call regex ensures it's not part of another word
                const regex = new RegExp(`(?:\\.${method}|\\b${method})\\s*\\(`, 'g');
                if (regex.test(line)) {
                    let file = null;
                    if (method.startsWith('read')) {
                        const fileMatch = line.match(/\((['"])(.*?)\1/);
                        if (fileMatch) file = fileMatch[2];
                    }
                    ops.push({
                        operation: method,
                        description: desc,
                        line: index + 1,
                        file: file
                    });
                }
            }
        });
        return ops;
    }

    extractInstantiations(content) {
        const instantiations = [];
        // Regex to find start of a call: name followed by (
        const callStartRegex = /([a-zA-Z0-9_.]+)\s*\(/g;

        let match;
        while ((match = callStartRegex.exec(content)) !== null) {
            const name = match[1];
            const startIndex = match.index + match[0].length;

            // Extract arguments by balancing parentheses
            let openParens = 1;
            let currentIndex = startIndex;
            let paramsStr = '';

            while (openParens > 0 && currentIndex < content.length) {
                const char = content[currentIndex];
                if (char === '(') openParens++;
                else if (char === ')') openParens--;

                if (openParens > 0) paramsStr += char;
                currentIndex++;
            }

            // Only add if we successfully closed the parens
            if (openParens === 0) {
                const params = this.parseParams(paramsStr);
                // Calculate line number based on newlines before the match
                const linesBefore = content.substring(0, match.index).split('\n').length;

                instantiations.push({
                    name,
                    line: linesBefore,
                    params
                });
            }
        }
        return instantiations;
    }

    parseParams(paramsStr) {
        const params = {};
        if (!paramsStr || !paramsStr.trim()) return params;

        // Naive split by comma, likely to fail on nested calls in args
        // Improving this requires a full parser, but for now we try to catch basic kwargs
        // We only care about kwargs like k=5 or C=1.0

        // Remove nested parentheses content to avoid splitting on commas inside them
        let cleanStr = '';
        let depth = 0;
        for (let char of paramsStr) {
            if (char === '(') depth++;
            if (char === ')') depth--;
            if (depth === 0) cleanStr += char;
            else cleanStr += ' '; // placeholder
        }

        const parts = cleanStr.split(',');
        parts.forEach(part => {
            const kv = part.split('=');
            if (kv.length === 2) {
                const key = kv[0].trim();
                // We don't try to capture the value perfectly if it's complex
                let val = kv[1].trim();
                // Try to parse basic types
                if (val === 'True') val = true;
                else if (val === 'False') val = false;
                else if (val === 'None') val = null;
                else if (!isNaN(val)) val = Number(val);
                else val = val.replace(/['"]/g, '');
                params[key] = val;
            }
        });

        return params;
    }

    estimateComplexity(modelData, params, dataOps, sourceModule = "") {
        const modelName = modelData.name;
        const info = this.modelInfo[modelName];

        let complexity, description;

        if (info) {
            complexity = {
                time_complexity: info.time,
                memory_complexity: info.memory,
                dataset_assumptions: info.use_case
            };
            description = {
                summary: info.summary,
                use_case: info.use_case,
                pros: info.pros,
                cons: info.cons
            };
        } else {
            // Heuristic Fallbacks based on Source Module or Name
            let heuristicTime = "Unknown";
            let heuristicMem = "Unknown";
            let heuristicSummary = `${modelData.type} from ${modelData.framework}`;

            if (sourceModule && sourceModule.includes("sklearn")) {
                if (sourceModule.includes("metrics") || modelName.includes("score") || modelName.includes("error")) {
                    heuristicTime = "O(n_samples)";
                    heuristicMem = "O(1)";
                    heuristicSummary = "Evaluation Metric";
                } else if (sourceModule.includes("preprocessing") || modelName.includes("Scaler") || modelName.includes("Encoder")) {
                    heuristicTime = "O(n_samples * n_features)";
                    heuristicMem = "O(n_features)";
                    heuristicSummary = "Data Preprocessing";
                } else if (sourceModule.includes("ensemble")) {
                    heuristicTime = "O(n_trees * n_samples * log(n_samples))";
                    heuristicMem = "O(n_trees * n_samples)";
                    heuristicSummary = "Ensemble Model";
                } else if (sourceModule.includes("linear_model")) {
                    heuristicTime = "O(n_samples * n_features^2)";
                    heuristicMem = "O(n_features)";
                    heuristicSummary = "Linear Model";
                }
            } else if (modelData.framework === "numpy") {
                heuristicTime = "O(n)";
                heuristicMem = "O(n)";
                heuristicSummary = "Array Operation";
            }

            complexity = {
                time_complexity: heuristicTime,
                memory_complexity: heuristicMem,
                dataset_assumptions: "Check documentation"
            };
            description = {
                summary: heuristicSummary,
                use_case: "General purpose",
                pros: ["Standard library function"],
                cons: []
            };
        }

        const fileAnalysis = this.analyzeFileContext(modelName, params, dataOps);

        return { complexity, description, file_analysis: fileAnalysis };
    }

    analyzeFileContext(modelName, params, dataOps) {
        const suggestions = [];
        let dataInfo = null;

        const csvFiles = dataOps.filter(op => op.operation === 'read_csv' && op.file).map(op => op.file);
        if (csvFiles.length > 0) {
            dataInfo = `Loading data from: ${csvFiles.join(', ')}`;
        }

        const hasSplit = dataOps.some(op => op.operation === 'train_test_split');
        if (!hasSplit && this.modelInfo[modelName]) {
            suggestions.push("Consider using train_test_split to evaluate model performance");
        }

        if (["LogisticRegression", "SVC", "SVR"].includes(modelName)) {
            if (!params.max_iter) suggestions.push("Consider setting max_iter if convergence warnings occur");
            if (!params.C) suggestions.push("Default C=1.0 is used. Tune C for better regularization");
        }

        if (["RandomForestClassifier", "RandomForestRegressor"].includes(modelName)) {
            const n_estimators = params.n_estimators || 100;
            if (n_estimators === 100) suggestions.push("Using default n_estimators=100. Consider increasing for better accuracy");
            if (!params.max_depth) suggestions.push("No max_depth set - trees may overfit. Consider limiting depth");
        }

        const needsScaling = ["LogisticRegression", "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor", "PCA"];
        const scalersUsed = dataOps.some(op => ['fit_transform', 'transform'].includes(op.operation));
        if (needsScaling.includes(modelName) && !scalersUsed) {
            suggestions.push("This model benefits from feature scaling (StandardScaler/MinMaxScaler)");
        }

        return {
            parameters_used: params,
            data_info: dataInfo,
            suggestions: suggestions,
            estimated_scale: dataOps.length > 0 ? "Data processing detected" : "Unknown"
        };
    }

    formatName(className) {
        return className.replace(/([A-Z])/g, ' $1').trim();
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLAnalyzer;
} else {
    window.MLAnalyzer = MLAnalyzer;
}
