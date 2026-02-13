from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ComplexityEstimate(BaseModel):
    time_complexity: str
    memory_complexity: str
    dataset_assumptions: str

class ModelDescription(BaseModel):
    """General description of what the model is"""
    summary: str  # What this model does
    use_case: str  # When to use it
    pros: List[str] = []
    cons: List[str] = []

class FileAnalysis(BaseModel):
    """Specific analysis based on the current file"""
    parameters_used: Dict[str, Any] = {}  # Actual params from code
    data_info: Optional[str] = None  # Dataset info if found
    suggestions: List[str] = []  # Recommendations for this usage
    estimated_scale: Optional[str] = None  # Small/Medium/Large based on context

class MLModelInfo(BaseModel):
    framework: str
    model_type: str
    model_name: str
    line_number: int
    complexity: Optional[ComplexityEstimate] = None
    description: Optional[ModelDescription] = None  # General info
    file_analysis: Optional[FileAnalysis] = None  # Specific to current file

class AnalysisResponse(BaseModel):
    filename: str
    detected_frameworks: List[str]
    models: List[MLModelInfo]
    pipeline_steps: List[str] = []
    data_operations: List[str] = []  # Data loading/processing found
    
class AnalysisRequest(BaseModel):
    repo_url: Optional[str] = None
    file_content: Optional[str] = None
    filename: str = "unknown.py"
