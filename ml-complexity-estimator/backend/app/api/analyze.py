from fastapi import APIRouter, HTTPException
from app.models.response_schema import (
    AnalysisRequest,
    AnalysisResponse,
    MLModelInfo
)
from app.core.ast_parser import ASTParser
from app.core.ml_detector import MLDetector
from app.core.complexity_estimator import ComplexityEstimator

router = APIRouter(tags=["Analysis"])

ml_detector = MLDetector()
complexity_estimator = ComplexityEstimator()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest):
    """
    Analyze Python code (from GitHub or Jupyter) and estimate
    ML model complexity using static AST analysis.
    """

    # ------------------ VALIDATION ------------------
    if not request.file_content or not request.file_content.strip():
        raise HTTPException(
            status_code=400,
            detail="file_content is required and cannot be empty"
        )

    # ------------------ PARSE AST ------------------
    try:
        tree = ASTParser.parse_code(request.file_content)
    except SyntaxError:
        raise HTTPException(
            status_code=400,
            detail="Invalid Python syntax"
        )

    if not tree:
        raise HTTPException(
            status_code=400,
            detail="Unable to parse Python code"
        )

    # ------------------ IMPORT DETECTION ------------------
    imports = ASTParser.get_imports(tree)
    imported_names = ASTParser.get_imported_names(tree)
    detected_frameworks = ml_detector.detect_frameworks(imports)

    # ------------------ DATA OPERATIONS ------------------
    data_ops = ASTParser.get_data_operations(tree)
    data_operations = [f"{op['description']} (line {op['line']})" for op in data_ops]

    # ------------------ MODEL DETECTION ------------------
    instantiations_with_params = ASTParser.get_class_instantiations_with_params(tree)
    
    # Build a map of class name to params for later lookup
    params_map = {}
    for item in instantiations_with_params:
        class_name = item['name'].split('.')[-1]
        params_map[class_name] = item['params']
    
    instantiations = [(item['name'], item['line']) for item in instantiations_with_params]
    raw_models = ml_detector.identify_models(
        instantiations,
        detected_frameworks,
        imported_names
    )

    models: list[MLModelInfo] = []

    for model in raw_models:
        # Get params for this specific model
        model_params = params_map.get(model["name"], {})
        
        # Get full analysis (complexity + description + file analysis)
        analysis = complexity_estimator.estimate(model, model_params, data_ops)

        models.append(
            MLModelInfo(
                framework=model["framework"],
                model_type=model["type"],
                model_name=model["name"],
                line_number=model["line"],
                complexity=analysis["complexity"],
                description=analysis["description"],
                file_analysis=analysis["file_analysis"]
            )
        )

    # ------------------ RESPONSE ------------------
    return AnalysisResponse(
        filename=request.filename or "unknown.py",
        detected_frameworks=detected_frameworks,
        models=models,
        data_operations=data_operations
    )
