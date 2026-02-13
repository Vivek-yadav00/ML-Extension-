from typing import List, Dict

class MLDetector:
    """
    Dynamically detects ML models based on import analysis.
    No hardcoded model names - detects ANY class from ML frameworks.
    """
    
    # ML framework patterns - maps framework name to module prefixes
    ML_FRAMEWORKS = {
        "sklearn": ["sklearn", "scikit-learn"],
        "tensorflow": ["tensorflow", "keras", "tf"],
        "pytorch": ["torch", "torchvision", "pytorch"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
        "catboost": ["catboost"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "scipy": ["scipy"],
        "statsmodels": ["statsmodels"],
        "transformers": ["transformers", "huggingface"],
    }
    
    # Module categories for complexity estimation
    MODULE_CATEGORIES = {
        # Sklearn submodules
        "linear_model": "linear",
        "tree": "tree",
        "ensemble": "ensemble",
        "svm": "svm",
        "neighbors": "neighbors",
        "cluster": "cluster",
        "preprocessing": "preprocessing",
        "decomposition": "decomposition",
        "neural_network": "neural_network",
        "naive_bayes": "naive_bayes",
        "metrics": "metrics",
        "model_selection": "model_selection",
        # Deep learning
        "nn": "neural_network",
        "layers": "neural_network",
        "models": "neural_network",
    }

    def detect_frameworks(self, imports: List[str]) -> List[str]:
        """Detect which ML frameworks are imported."""
        detected = set()
        for imp in imports:
            imp_lower = imp.lower()
            for framework, prefixes in self.ML_FRAMEWORKS.items():
                if any(imp_lower.startswith(p) or p in imp_lower for p in prefixes):
                    detected.add(framework)
        return list(detected)

    def identify_models(
        self, 
        instantiations: List[tuple], 
        frameworks: List[str],
        imported_names: Dict[str, str] = None
    ) -> List[Dict]:
        """
        Dynamically identify ML models by matching instantiations 
        against imported names from ML frameworks.
        """
        models = []
        imported_names = imported_names or {}
        
        for call_name, lineno in instantiations:
            # Get the class name (last part of dotted name)
            class_name = call_name.split(".")[-1]
            
            # Skip lowercase names (likely functions, not classes)
            if class_name[0].islower():
                continue
            
            # Check if this name was imported from an ML framework
            source_module = imported_names.get(class_name, "")
            
            # Also check if it's a dotted call like sklearn.linear_model.LogisticRegression()
            if not source_module:
                for part in call_name.split("."):
                    for framework, prefixes in self.ML_FRAMEWORKS.items():
                        if any(part.lower().startswith(p) for p in prefixes):
                            source_module = call_name.rsplit(".", 1)[0] if "." in call_name else framework
                            break
            
            if not source_module:
                continue
                
            # Determine the framework
            framework = self._get_framework(source_module)
            if not framework:
                continue
            
            # Determine the category/type based on module path
            category = self._get_category(source_module)
            
            # Create human-readable model type
            model_type = self._format_model_name(class_name)
            
            models.append({
                "name": class_name,
                "type": model_type,
                "framework": framework,
                "module": source_module,
                "category": category,
                "line": lineno
            })
        
        # Remove duplicates (same model on same line)
        seen = set()
        unique_models = []
        for m in models:
            key = (m["name"], m["line"])
            if key not in seen:
                seen.add(key)
                unique_models.append(m)
        
        return unique_models
    
    def _get_framework(self, module: str) -> str:
        """Determine which framework a module belongs to."""
        module_lower = module.lower()
        for framework, prefixes in self.ML_FRAMEWORKS.items():
            if any(module_lower.startswith(p) or f".{p}" in module_lower for p in prefixes):
                return framework
        return None
    
    def _get_category(self, module: str) -> str:
        """Determine the category of the model based on module path."""
        module_lower = module.lower()
        for key, category in self.MODULE_CATEGORIES.items():
            if key in module_lower:
                return category
        return "general"
    
    def _format_model_name(self, class_name: str) -> str:
        """Convert CamelCase class name to readable format."""
        # Insert spaces before capitals
        result = ""
        for i, char in enumerate(class_name):
            if char.isupper() and i > 0:
                # Don't add space if previous char was also uppercase (acronyms)
                if not class_name[i-1].isupper():
                    result += " "
                # Add space if next char is lowercase (end of acronym)
                elif i + 1 < len(class_name) and class_name[i+1].islower():
                    result += " "
            result += char
        return result
