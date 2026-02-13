from typing import Dict, Any, List
from app.models.response_schema import ComplexityEstimate, ModelDescription, FileAnalysis

class ComplexityEstimator:
    """
    Provides both general model descriptions and file-specific analysis.
    """
    
    # General descriptions for ML models
    MODEL_INFO = {
        # Linear Models
        "LogisticRegression": {
            "summary": "A linear classifier that predicts probabilities using the logistic function. Despite its name, it's used for classification, not regression.",
            "use_case": "Binary/multi-class classification with linearly separable data",
            "pros": ["Fast training and prediction", "Interpretable coefficients", "Works well with small datasets", "Provides probability estimates"],
            "cons": ["Assumes linear decision boundary", "May underfit complex patterns", "Sensitive to feature scaling"],
            "time": "O(n_samples * n_features * n_classes)",
            "memory": "O(n_features * n_classes)",
        },
        "LinearRegression": {
            "summary": "Fits a linear model to minimize the sum of squared residuals between observed and predicted values.",
            "use_case": "Predicting continuous values with linear relationships",
            "pros": ["Very fast", "Simple and interpretable", "No hyperparameters to tune", "Analytical solution"],
            "cons": ["Assumes linear relationship", "Sensitive to outliers", "Cannot capture complex patterns"],
            "time": "O(n_samples * n_features²)",
            "memory": "O(n_features)",
        },
        "Ridge": {
            "summary": "Linear regression with L2 regularization to prevent overfitting by penalizing large coefficients.",
            "use_case": "Regression when features are correlated or when you have more features than samples",
            "pros": ["Prevents overfitting", "Handles multicollinearity", "More stable than OLS"],
            "cons": ["Doesn't perform feature selection", "Requires tuning alpha parameter"],
            "time": "O(n_samples * n_features²)",
            "memory": "O(n_features)",
        },
        "Lasso": {
            "summary": "Linear regression with L1 regularization that can shrink some coefficients to zero, performing feature selection.",
            "use_case": "Regression with automatic feature selection",
            "pros": ["Performs feature selection", "Creates sparse models", "Good for high-dimensional data"],
            "cons": ["Can be unstable with correlated features", "May select only one from correlated features"],
            "time": "O(n_samples * n_features²)",
            "memory": "O(n_features)",
        },
        
        # Tree-based Models
        "DecisionTreeClassifier": {
            "summary": "A tree-based classifier that splits data based on feature values to make predictions.",
            "use_case": "Classification with interpretable decision rules",
            "pros": ["Easy to understand and visualize", "Handles non-linear relationships", "No feature scaling needed"],
            "cons": ["Prone to overfitting", "Unstable (small changes affect tree)", "Biased towards features with more levels"],
            "time": "O(n_samples * n_features * log(n_samples))",
            "memory": "O(n_nodes)",
        },
        "DecisionTreeRegressor": {
            "summary": "A tree-based regressor that predicts continuous values by averaging target values in leaf nodes.",
            "use_case": "Regression with non-linear relationships",
            "pros": ["Captures non-linear patterns", "No feature scaling needed", "Interpretable"],
            "cons": ["Prone to overfitting", "Creates step-wise predictions", "High variance"],
            "time": "O(n_samples * n_features * log(n_samples))",
            "memory": "O(n_nodes)",
        },
        "RandomForestClassifier": {
            "summary": "An ensemble of decision trees trained on random subsets of data and features, using majority voting.",
            "use_case": "General-purpose classification with good accuracy",
            "pros": ["Reduces overfitting vs single tree", "Handles high-dimensional data", "Provides feature importance"],
            "cons": ["Less interpretable than single tree", "Can be slow for large datasets", "Memory intensive"],
            "time": "O(n_trees * n_samples * log(n_samples) * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "RandomForestRegressor": {
            "summary": "An ensemble of decision trees for regression, averaging predictions from multiple trees.",
            "use_case": "General-purpose regression with robustness to outliers",
            "pros": ["Good accuracy out-of-the-box", "Robust to outliers", "Provides feature importance"],
            "cons": ["Cannot extrapolate beyond training data", "Memory intensive", "Slower prediction than linear models"],
            "time": "O(n_trees * n_samples * log(n_samples) * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "GradientBoostingClassifier": {
            "summary": "Builds an ensemble of trees sequentially, where each tree corrects errors of the previous ones.",
            "use_case": "High-accuracy classification when training time is not critical",
            "pros": ["Often achieves best accuracy", "Handles mixed feature types", "Feature importance"],
            "cons": ["Sequential training (not parallelizable)", "Prone to overfitting without tuning", "Slow training"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "GradientBoostingRegressor": {
            "summary": "Sequential ensemble of trees for regression, each tree fitting the residual errors.",
            "use_case": "High-accuracy regression problems",
            "pros": ["Excellent predictive accuracy", "Handles non-linear relationships"],
            "cons": ["Slow training", "Requires careful hyperparameter tuning"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        
        # SVM
        "SVC": {
            "summary": "Support Vector Classifier finds the optimal hyperplane that maximizes margin between classes.",
            "use_case": "Binary classification with clear margin of separation",
            "pros": ["Effective in high dimensions", "Works well with clear margins", "Kernel trick for non-linear data"],
            "cons": ["Scales poorly O(n²) to O(n³)", "Sensitive to feature scaling", "Not suitable for >10k samples"],
            "time": "O(n_samples² * n_features) to O(n_samples³)",
            "memory": "O(n_samples * n_features)",
        },
        "SVR": {
            "summary": "Support Vector Regression predicts continuous values while trying to fit data within an epsilon-tube.",
            "use_case": "Regression when you want to ignore small errors",
            "pros": ["Robust to outliers", "Works well in high dimensions"],
            "cons": ["Very slow for large datasets", "Sensitive to hyperparameters"],
            "time": "O(n_samples² * n_features)",
            "memory": "O(n_samples * n_features)",
        },
        
        # Neighbors
        "KNeighborsClassifier": {
            "summary": "Classifies based on majority vote of k nearest neighbors in feature space.",
            "use_case": "Classification with local patterns, small datasets",
            "pros": ["Simple and intuitive", "No training time", "Adapts to any shape of decision boundary"],
            "cons": ["Slow prediction (checks all points)", "Curse of dimensionality", "Requires feature scaling"],
            "time": "O(1) training, O(n_samples * n_features) prediction",
            "memory": "O(n_samples * n_features)",
        },
        "KNeighborsRegressor": {
            "summary": "Predicts by averaging values of k nearest neighbors.",
            "use_case": "Regression with local patterns",
            "pros": ["Simple", "No assumptions about data distribution"],
            "cons": ["Memory intensive", "Slow predictions for large data"],
            "time": "O(1) training, O(n_samples * n_features) prediction",
            "memory": "O(n_samples * n_features)",
        },
        
        # Naive Bayes
        "GaussianNB": {
            "summary": "Probabilistic classifier assuming features follow Gaussian distribution and are independent.",
            "use_case": "Text classification, spam detection, quick baseline",
            "pros": ["Extremely fast", "Works well with small data", "Handles high dimensions"],
            "cons": ["Assumes feature independence (rarely true)", "Can be outperformed by other methods"],
            "time": "O(n_samples * n_features)",
            "memory": "O(n_classes * n_features)",
        },
        "MultinomialNB": {
            "summary": "Naive Bayes for discrete features like word counts in text classification.",
            "use_case": "Text classification, document categorization",
            "pros": ["Fast", "Works well with text data", "Handles high dimensions"],
            "cons": ["Assumes feature independence", "Only for discrete features"],
            "time": "O(n_samples * n_features)",
            "memory": "O(n_classes * n_features)",
        },
        
        # Clustering
        "KMeans": {
            "summary": "Partitions data into k clusters by minimizing within-cluster variance iteratively.",
            "use_case": "Customer segmentation, data compression, pattern discovery",
            "pros": ["Simple and fast", "Scales well", "Easy to interpret"],
            "cons": ["Must specify k in advance", "Sensitive to initialization", "Assumes spherical clusters"],
            "time": "O(n_samples * k * n_iterations * n_features)",
            "memory": "O(n_samples * n_features + k * n_features)",
        },
        "DBSCAN": {
            "summary": "Density-based clustering that finds clusters of arbitrary shape and identifies outliers.",
            "use_case": "Finding clusters of arbitrary shape, outlier detection",
            "pros": ["No need to specify cluster count", "Finds arbitrary shapes", "Identifies outliers"],
            "cons": ["Sensitive to eps and min_samples", "Struggles with varying densities"],
            "time": "O(n_samples²) worst case, O(n * log n) with spatial index",
            "memory": "O(n_samples)",
        },
        
        # Preprocessing
        "StandardScaler": {
            "summary": "Standardizes features by removing mean and scaling to unit variance (z-score normalization).",
            "use_case": "Feature scaling before SVM, KNN, neural networks, or any distance-based algorithm",
            "pros": ["Makes features comparable", "Required by many algorithms", "Handles different scales"],
            "cons": ["Sensitive to outliers", "Changes data distribution"],
            "time": "O(n_samples * n_features)",
            "memory": "O(n_features)",
        },
        "MinMaxScaler": {
            "summary": "Scales features to a fixed range [0, 1] based on min and max values.",
            "use_case": "When you need bounded values, neural networks",
            "pros": ["Preserves zero values in sparse data", "Bounded output range"],
            "cons": ["Very sensitive to outliers", "Affected by min/max values"],
            "time": "O(n_samples * n_features)",
            "memory": "O(n_features)",
        },
        "LabelEncoder": {
            "summary": "Encodes categorical labels as integers (0, 1, 2, ...).",
            "use_case": "Converting categorical target variable to numbers",
            "pros": ["Simple", "Memory efficient"],
            "cons": ["Implies ordinal relationship", "Use OneHotEncoder for features"],
            "time": "O(n_samples)",
            "memory": "O(n_classes)",
        },
        "OneHotEncoder": {
            "summary": "Converts categorical features to binary vectors (one column per category).",
            "use_case": "Encoding categorical features for ML models",
            "pros": ["No ordinal assumption", "Works with tree and linear models"],
            "cons": ["Increases dimensionality", "Can cause sparse matrices"],
            "time": "O(n_samples * n_categories)",
            "memory": "O(n_samples * n_categories)",
        },
        
        # Dimensionality Reduction
        "PCA": {
            "summary": "Principal Component Analysis reduces dimensions by projecting data onto directions of maximum variance.",
            "use_case": "Dimensionality reduction, visualization, noise reduction",
            "pros": ["Reduces overfitting", "Speeds up training", "Removes collinearity"],
            "cons": ["Loses interpretability", "Assumes linear relationships", "Sensitive to scaling"],
            "time": "O(n_features² * n_samples + n_features³)",
            "memory": "O(n_features²)",
        },
        
        # XGBoost/LightGBM
        "XGBClassifier": {
            "summary": "Extreme Gradient Boosting - optimized gradient boosting with regularization.",
            "use_case": "Competitions, high-accuracy classification",
            "pros": ["State-of-the-art accuracy", "Handles missing values", "Fast and parallelizable"],
            "cons": ["Many hyperparameters", "Can overfit without tuning"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "XGBRegressor": {
            "summary": "XGBoost for regression tasks with gradient boosting.",
            "use_case": "High-accuracy regression",
            "pros": ["Excellent accuracy", "Handles missing data", "Feature importance"],
            "cons": ["Black box", "Tuning required"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "LGBMClassifier": {
            "summary": "Light Gradient Boosting Machine - faster and more memory-efficient gradient boosting.",
            "use_case": "Large datasets, fast training needed",
            "pros": ["Faster than XGBoost", "Lower memory usage", "Handles large data"],
            "cons": ["Can overfit on small data", "Sensitive to hyperparameters"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
        "LGBMRegressor": {
            "summary": "LightGBM for regression with leaf-wise tree growth.",
            "use_case": "Large-scale regression",
            "pros": ["Very fast", "Memory efficient", "Good accuracy"],
            "cons": ["Requires careful tuning"],
            "time": "O(n_trees * n_samples * n_features)",
            "memory": "O(n_trees * n_leaves)",
        },
    }

    def estimate(self, model_info: dict, params: dict = None, data_ops: list = None) -> dict:
        """
        Returns complexity, general description, and file-specific analysis.
        """
        model_name = model_info.get("name", "")
        model_type = model_info.get("type", "")
        framework = model_info.get("framework", "")
        params = params or {}
        data_ops = data_ops or []
        
        # Get general model info
        info = self.MODEL_INFO.get(model_name, None)
        
        # Build complexity estimate
        if info:
            complexity = ComplexityEstimate(
                time_complexity=info.get("time", "Unknown"),
                memory_complexity=info.get("memory", "Unknown"),
                dataset_assumptions=info.get("use_case", "N/A")
            )
            description = ModelDescription(
                summary=info.get("summary", "ML model"),
                use_case=info.get("use_case", "General purpose"),
                pros=info.get("pros", []),
                cons=info.get("cons", [])
            )
        else:
            complexity = ComplexityEstimate(
                time_complexity="Unknown",
                memory_complexity="Unknown",
                dataset_assumptions="N/A"
            )
            description = ModelDescription(
                summary=f"{model_type} model from {framework} framework",
                use_case="Refer to official documentation",
                pros=[],
                cons=[]
            )
        
        # Build file-specific analysis
        file_analysis = self._analyze_file_context(model_name, params, data_ops)
        
        return {
            "complexity": complexity,
            "description": description,
            "file_analysis": file_analysis
        }
    
    def _analyze_file_context(self, model_name: str, params: dict, data_ops: list) -> FileAnalysis:
        """Analyze the specific usage in the current file."""
        suggestions = []
        estimated_scale = "Unknown"
        data_info = None
        
        # Analyze data operations
        csv_files = [op.get('file') for op in data_ops if op.get('operation') == 'read_csv' and op.get('file')]
        if csv_files:
            data_info = f"Loading data from: {', '.join(str(f) for f in csv_files)}"
        
        # Check for train_test_split
        has_split = any(op.get('operation') == 'train_test_split' for op in data_ops)
        if not has_split and model_name in self.MODEL_INFO:
            suggestions.append("Consider using train_test_split to evaluate model performance")
        
        # Analyze parameters and give suggestions
        if model_name in ["LogisticRegression", "SVC", "SVR"]:
            if "max_iter" not in params:
                suggestions.append("Consider setting max_iter if convergence warnings occur")
            if "C" not in params:
                suggestions.append("Default C=1.0 is used. Tune C for better regularization")
        
        if model_name in ["RandomForestClassifier", "RandomForestRegressor"]:
            n_estimators = params.get("n_estimators", 100)
            if n_estimators == 100:
                suggestions.append("Using default n_estimators=100. Consider increasing for better accuracy")
            if "max_depth" not in params:
                suggestions.append("No max_depth set - trees may overfit. Consider limiting depth")
        
        if model_name in ["KMeans"]:
            if "n_clusters" not in params:
                suggestions.append("Default n_clusters=8 used. Use elbow method to find optimal k")
        
        if model_name in ["KNeighborsClassifier", "KNeighborsRegressor"]:
            n_neighbors = params.get("n_neighbors", 5)
            suggestions.append(f"Using k={n_neighbors} neighbors. Consider cross-validation to tune k")
        
        # Check for scaling with models that need it
        needs_scaling = ["LogisticRegression", "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor", "PCA"]
        scalers_used = any(op.get('operation') in ['fit_transform', 'transform'] for op in data_ops)
        if model_name in needs_scaling and not scalers_used:
            suggestions.append("This model benefits from feature scaling (StandardScaler/MinMaxScaler)")
        
        # Estimate scale based on operations
        if data_ops:
            estimated_scale = "Data processing detected"
        
        return FileAnalysis(
            parameters_used=params,
            data_info=data_info,
            suggestions=suggestions,
            estimated_scale=estimated_scale
        )
