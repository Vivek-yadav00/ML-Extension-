import ast
from typing import List, Tuple, Dict, Any

class ASTParser:
    @staticmethod
    def parse_code(code: str) -> ast.AST:
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    @staticmethod
    def get_imports(tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return list(set(imports))

    @staticmethod
    def get_imported_names(tree: ast.AST) -> Dict[str, str]:
        """
        Returns a mapping of imported name -> full module path
        e.g., {'LogisticRegression': 'sklearn.linear_model', 'pd': 'pandas'}
        """
        imported_names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imported_names[name] = node.module
        return imported_names

    @staticmethod
    def get_class_instantiations_with_params(tree: ast.AST) -> List[Dict]:
        """
        Get class instantiations with their parameters.
        Returns: [{'name': 'LogisticRegression', 'line': 5, 'params': {'C': 0.1, 'max_iter': 100}}]
        """
        instantiations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    parts = []
                    curr = node.func
                    while isinstance(curr, ast.Attribute):
                        parts.append(curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        parts.append(curr.id)
                    name = ".".join(reversed(parts))
                
                if name:
                    # Extract parameters
                    params = ASTParser._extract_call_params(node)
                    instantiations.append({
                        'name': name,
                        'line': node.lineno,
                        'params': params
                    })
        return instantiations

    @staticmethod
    def get_class_instantiations(tree: ast.AST) -> List[Tuple[str, int]]:
        """Legacy method for backward compatibility"""
        result = ASTParser.get_class_instantiations_with_params(tree)
        return [(item['name'], item['line']) for item in result]

    @staticmethod
    def _extract_call_params(node: ast.Call) -> Dict[str, Any]:
        """Extract keyword arguments from a function call."""
        params = {}
        for kw in node.keywords:
            if kw.arg:
                params[kw.arg] = ASTParser._get_value(kw.value)
        return params

    @staticmethod
    def _get_value(node: ast.expr) -> Any:
        """Extract Python value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id  # Variable name
        elif isinstance(node, ast.List):
            return [ASTParser._get_value(e) for e in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(ASTParser._get_value(e) for e in node.elts)
        elif isinstance(node, ast.Dict):
            return {ASTParser._get_value(k): ASTParser._get_value(v) 
                    for k, v in zip(node.keys, node.values) if k}
        elif isinstance(node, ast.NameConstant):  # True, False, None
            return node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ASTParser._get_value(node.operand)
        else:
            return "<expression>"

    @staticmethod
    def get_data_operations(tree: ast.AST) -> List[Dict]:
        """
        Find data loading and processing operations in the code.
        E.g., pd.read_csv(), train_test_split(), etc.
        """
        data_ops = []
        data_methods = {
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
            'shape': 'Data dimensions',
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method_name = ""
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    method_name = node.func.id
                
                if method_name in data_methods:
                    # Try to extract filename for read operations
                    file_arg = None
                    if node.args and method_name.startswith('read'):
                        file_arg = ASTParser._get_value(node.args[0])
                    
                    data_ops.append({
                        'operation': method_name,
                        'description': data_methods[method_name],
                        'line': node.lineno,
                        'file': file_arg
                    })
        
        return data_ops

    @staticmethod
    def find_variable_assignment(tree: ast.AST, var_name: str) -> Any:
        """Find what value was assigned to a variable."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return ASTParser._get_value(node.value)
        return None
