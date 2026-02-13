from typing import List, Dict

class MLDetector:
    """
    Detects ML models from Python code using import analysis and instantiation detection.
    """
    
    FRAMEWORKS = {
        "sklearn": ["sklearn", "scikit"],
        "tensorflow": ["tensorflow", "keras", "tf"],
        "pytorch": ["torch", "torchvision"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
        "catboost": ["catboost"],
    }

    # Known ML model classes and their human-readable names
    KNOWN_MODELS = {
        # Scikit-learn - Classification
        "LogisticRegression": ("Logistic Regression", "sklearn"),
        "SVC": ("Support Vector Classifier", "sklearn"),
        "SVR": ("Support Vector Regressor", "sklearn"),
        "RandomForestClassifier": ("Random Forest Classifier", "sklearn"),
        "RandomForestRegressor": ("Random Forest Regressor", "sklearn"),
        "GradientBoostingClassifier": ("Gradient Boosting Classifier", "sklearn"),
        "GradientBoostingRegressor": ("Gradient Boosting Regressor", "sklearn"),
        "DecisionTreeClassifier": ("Decision Tree Classifier", "sklearn"),
        "DecisionTreeRegressor": ("Decision Tree Regressor", "sklearn"),
        "KNeighborsClassifier": ("K-Nearest Neighbors Classifier", "sklearn"),
        "KNeighborsRegressor": ("K-Nearest Neighbors Regressor", "sklearn"),
        "AdaBoostClassifier": ("AdaBoost Classifier", "sklearn"),
        "AdaBoostRegressor": ("AdaBoost Regressor", "sklearn"),
        "ExtraTreesClassifier": ("Extra Trees Classifier", "sklearn"),
        "ExtraTreesRegressor": ("Extra Trees Regressor", "sklearn"),
        "GaussianNB": ("Gaussian Naive Bayes", "sklearn"),
        "MultinomialNB": ("Multinomial Naive Bayes", "sklearn"),
        "BernoulliNB": ("Bernoulli Naive Bayes", "sklearn"),
        # Regression
        "LinearRegression": ("Linear Regression", "sklearn"),
        "Ridge": ("Ridge Regression", "sklearn"),
        "Lasso": ("Lasso Regression", "sklearn"),
        "ElasticNet": ("Elastic Net", "sklearn"),
        # Clustering
        "KMeans": ("K-Means Clustering", "sklearn"),
        "DBSCAN": ("DBSCAN Clustering", "sklearn"),
        "AgglomerativeClustering": ("Agglomerative Clustering", "sklearn"),
        # Preprocessing
        "StandardScaler": ("Standard Scaler", "sklearn"),
        "MinMaxScaler": ("MinMax Scaler", "sklearn"),
        "LabelEncoder": ("Label Encoder", "sklearn"),
        "OneHotEncoder": ("One-Hot Encoder", "sklearn"),
        "PCA": ("Principal Component Analysis", "sklearn"),
        # Ensemble
        "VotingClassifier": ("Voting Classifier", "sklearn"),
        "BaggingClassifier": ("Bagging Classifier", "sklearn"),
        "StackingClassifier": ("Stacking Classifier", "sklearn"),
        # XGBoost
        "XGBClassifier": ("XGBoost Classifier", "xgboost"),
        "XGBRegressor": ("XGBoost Regressor", "xgboost"),
        # LightGBM
        "LGBMClassifier": ("LightGBM Classifier", "lightgbm"),
        "LGBMRegressor": ("LightGBM Regressor", "lightgbm"),
        # CatBoost
        "CatBoostClassifier": ("CatBoost Classifier", "catboost"),
        "CatBoostRegressor": ("CatBoost Regressor", "catboost"),
        # PyTorch
        "Linear": ("Linear Layer", "pytorch"),
        "Conv2d": ("Convolutional Layer", "pytorch"),
        "Conv1d": ("1D Convolutional Layer", "pytorch"),
        "LSTM": ("LSTM Layer", "pytorch"),
        "GRU": ("GRU Layer", "pytorch"),
        "Sequential": ("Sequential Model", "pytorch"),
        "Module": ("Neural Network Module", "pytorch"),
        # TensorFlow/Keras
        "Dense": ("Dense Layer", "tensorflow"),
        "Conv2D": ("Convolutional Layer", "tensorflow"),
        "LSTM": ("LSTM Layer", "tensorflow"),
        "Dropout": ("Dropout Layer", "tensorflow"),
        "BatchNormalization": ("Batch Normalization", "tensorflow"),
    }

    def detect_frameworks(self, imports: List[str]) -> List[str]:
        """Detect which ML frameworks are imported."""
        detected = set()
        for imp in imports:
            imp_lower = imp.lower()
            for framework, keywords in self.FRAMEWORKS.items():
                if any(k in imp_lower for k in keywords):
                    detected.add(framework)
        return list(detected)

    def identify_models(
        self, 
        instantiations: List[tuple], 
        frameworks: List[str],
        imported_names: Dict[str, str] = None
    ) -> List[Dict]:
        """
        Identify ML models by matching instantiations against:
        1. Known model classes
        2. Dynamically imported classes from ML frameworks
        """
        models = []
        imported_names = imported_names or {}
        seen = set()  # Avoid duplicates

        for name, lineno in instantiations:
            # Get the class name (last part of dotted name)
            class_name = name.split(".")[-1]
            
            # Check if this is a known ML model
            if class_name in self.KNOWN_MODELS:
                model_type, framework = self.KNOWN_MODELS[class_name]
                key = (class_name, lineno)
                if key not in seen:
                    seen.add(key)
                    models.append({
                        "name": class_name,
                        "type": model_type,
                        "framework": framework,
                        "line": lineno
                    })
                continue
            
            # Check if imported from ML framework (dynamic detection)
            source_module = imported_names.get(class_name, "")
            if source_module:
                for framework, keywords in self.FRAMEWORKS.items():
                    if any(k in source_module.lower() for k in keywords):
                        # It's from an ML framework - add it
                        model_type = self._format_name(class_name)
                        key = (class_name, lineno)
                        if key not in seen:
                            seen.add(key)
                            models.append({
                                "name": class_name,
                                "type": model_type,
                                "framework": framework,
                                "line": lineno
                            })
                        break
        
        return models
    
    def _format_name(self, class_name: str) -> str:
        """Convert CamelCase to readable format."""
        result = ""
        for i, char in enumerate(class_name):
            if char.isupper() and i > 0:
                if not class_name[i-1].isupper():
                    result += " "
                elif i + 1 < len(class_name) and class_name[i+1].islower():
                    result += " "
            result += char
        return result
