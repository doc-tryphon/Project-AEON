#!/usr/bin/env python3
"""
Comprehensive Codebase Analysis Tool for Quantum Wormhole Simulator

This script performs detailed analysis of the entire codebase, generating
metrics, dependency analysis, and technical assessment for the SITREP.
"""

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import subprocess

class CodebaseAnalyzer:
    """Comprehensive analysis of the quantum wormhole simulation codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.analysis_results = {}
        
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze overall file structure and organization."""
        
        structure = {
            'total_files': 0,
            'python_files': 0,
            'source_files': 0,
            'test_files': 0,
            'config_files': 0,
            'documentation_files': 0,
            'directories': {},
            'file_types': Counter(),
            'largest_files': []
        }
        
        # Walk through all files
        for root, dirs, files in os.walk(self.project_root):
            rel_root = os.path.relpath(root, self.project_root)
            structure['directories'][rel_root] = len(files)
            
            for file in files:
                full_path = Path(root) / file
                structure['total_files'] += 1
                
                # Categorize by extension
                ext = file.split('.')[-1] if '.' in file else 'no_ext'
                structure['file_types'][ext] += 1
                
                # Categorize by type
                if file.endswith('.py'):
                    structure['python_files'] += 1
                    if 'src/' in str(full_path):
                        structure['source_files'] += 1
                    elif 'test' in str(full_path).lower():
                        structure['test_files'] += 1
                        
                elif file.endswith(('.json', '.yaml', '.yml', '.toml')):
                    structure['config_files'] += 1
                elif file.endswith(('.md', '.rst', '.txt')):
                    structure['documentation_files'] += 1
                
                # Track file sizes
                try:
                    size = full_path.stat().st_size
                    if file.endswith('.py'):
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.readlines())
                        structure['largest_files'].append((str(full_path), lines, size))
                except:
                    pass
        
        # Sort largest files
        structure['largest_files'].sort(key=lambda x: x[1], reverse=True)
        structure['largest_files'] = structure['largest_files'][:10]
        
        return structure
    
    def analyze_source_code_metrics(self) -> Dict[str, Any]:
        """Analyze source code metrics like complexity, dependencies, etc."""
        
        metrics = {
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0,
            'complexity_scores': [],
            'module_analysis': {},
            'dependency_map': defaultdict(set),
            'circular_dependencies': [],
            'error_handling_coverage': 0,
            'docstring_coverage': 0
        }
        
        # Analyze each Python file in src/
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Parse AST
                try:
                    tree = ast.parse(content)
                    file_metrics = self._analyze_ast(tree, py_file, content)
                    
                    module_name = str(py_file.relative_to(self.src_dir)).replace('/', '.').replace('\\', '.')[:-3]
                    metrics['module_analysis'][module_name] = file_metrics
                    
                    # Accumulate totals
                    metrics['total_lines'] += len(lines)
                    metrics['total_functions'] += file_metrics['function_count']
                    metrics['total_classes'] += file_metrics['class_count']
                    metrics['total_imports'] += file_metrics['import_count']
                    
                    if file_metrics['complexity_score']:
                        metrics['complexity_scores'].extend(file_metrics['complexity_score'])
                    
                    # Build dependency map
                    for imp in file_metrics['imports']:
                        if imp.startswith('src.'):
                            metrics['dependency_map'][module_name].add(imp)
                    
                except SyntaxError:
                    metrics['module_analysis'][str(py_file)] = {'error': 'Syntax error'}
                    
            except Exception as e:
                metrics['module_analysis'][str(py_file)] = {'error': str(e)}
        
        # Calculate averages
        if metrics['complexity_scores']:
            metrics['avg_complexity'] = sum(metrics['complexity_scores']) / len(metrics['complexity_scores'])
            metrics['max_complexity'] = max(metrics['complexity_scores'])
        
        # Calculate coverage percentages
        total_modules = len([m for m in metrics['module_analysis'] if 'error' not in metrics['module_analysis'][m]])
        if total_modules > 0:
            error_handling_modules = sum(1 for m in metrics['module_analysis'].values() 
                                       if 'error' not in m and m.get('has_error_handling', False))
            docstring_modules = sum(1 for m in metrics['module_analysis'].values() 
                                  if 'error' not in m and m.get('docstring_coverage', 0) > 0.5)
            
            metrics['error_handling_coverage'] = error_handling_modules / total_modules
            metrics['docstring_coverage'] = docstring_modules / total_modules
        
        return metrics
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze a single file's AST for detailed metrics."""
        
        analysis = {
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'imports': [],
            'complexity_score': [],
            'has_error_handling': False,
            'docstring_coverage': 0,
            'type_hints_coverage': 0,
            'test_functions': 0
        }
        
        functions_with_docstrings = 0
        functions_with_type_hints = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['function_count'] += 1
                total_functions += 1
                
                # Check for test functions
                if node.name.startswith('test_'):
                    analysis['test_functions'] += 1
                
                # Check docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    functions_with_docstrings += 1
                
                # Check type hints
                if (node.returns or any(arg.annotation for arg in node.args.args)):
                    functions_with_type_hints += 1
                
                # Calculate cyclomatic complexity (simplified)
                complexity = self._calculate_complexity(node)
                analysis['complexity_score'].append(complexity)
                
            elif isinstance(node, ast.ClassDef):
                analysis['class_count'] += 1
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                analysis['import_count'] += 1
                
                if isinstance(node, ast.ImportFrom) and node.module:
                    analysis['imports'].append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
            
            elif isinstance(node, (ast.Try, ast.ExceptHandler, ast.Raise)):
                analysis['has_error_handling'] = True
        
        # Calculate coverage percentages
        if total_functions > 0:
            analysis['docstring_coverage'] = functions_with_docstrings / total_functions
            analysis['type_hints_coverage'] = functions_with_type_hints / total_functions
        
        return analysis
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function (simplified)."""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze external and internal dependencies."""
        
        deps = {
            'external_packages': Counter(),
            'internal_modules': Counter(),
            'dependency_graph': {},
            'missing_dependencies': [],
            'version_constraints': {}
        }
        
        # Parse requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '>=' in line:
                            pkg, version = line.split('>=')
                            deps['version_constraints'][pkg] = version.split(',')[0]
                        elif '==' in line:
                            pkg, version = line.split('==')
                            deps['version_constraints'][pkg] = version
                        else:
                            pkg = line.split('<')[0].split('>')[0]
                            deps['version_constraints'][pkg] = 'any'
        
        # Analyze imports in source files
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find all imports
                imports = re.findall(r'^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content, re.MULTILINE)
                
                for imp in imports:
                    if imp.startswith('src.'):
                        deps['internal_modules'][imp] += 1
                    else:
                        # Extract top-level package
                        top_level = imp.split('.')[0]
                        if top_level not in ['os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random']:
                            deps['external_packages'][top_level] += 1
                            
            except Exception:
                pass
        
        return deps
    
    def analyze_physics_components(self) -> Dict[str, Any]:
        """Analyze physics implementation components."""
        
        physics_analysis = {
            'core_modules': {},
            'physics_coverage': {},
            'numerical_methods': [],
            'validation_status': {},
            'known_issues': []
        }
        
        physics_dir = self.src_dir / "physics"
        if physics_dir.exists():
            for py_file in physics_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                module_name = py_file.stem
                analysis = self._analyze_physics_module(py_file)
                physics_analysis['core_modules'][module_name] = analysis
        
        return physics_analysis
    
    def _analyze_physics_module(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a specific physics module."""
        
        analysis = {
            'line_count': 0,
            'class_count': 0,
            'function_count': 0,
            'has_tests': False,
            'physics_concepts': [],
            'numerical_libraries': [],
            'complexity_rating': 'medium'
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                analysis['line_count'] = len(lines)
            
            # Look for physics-related keywords
            physics_keywords = [
                'metric', 'tensor', 'einstein', 'schwarzschild', 'morris', 'thorne',
                'spacetime', 'curvature', 'geodesic', 'exotic', 'matter', 'energy',
                'christoffel', 'ricci', 'weyl', 'stress_energy', 'field_equations'
            ]
            
            for keyword in physics_keywords:
                if keyword.lower() in content.lower():
                    analysis['physics_concepts'].append(keyword)
            
            # Look for numerical libraries
            numerical_libs = ['numpy', 'scipy', 'sympy', 'matplotlib']
            for lib in numerical_libs:
                if lib in content:
                    analysis['numerical_libraries'].append(lib)
            
            # Parse AST for structure
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        analysis['class_count'] += 1
                    elif isinstance(node, ast.FunctionDef):
                        analysis['function_count'] += 1
            except:
                pass
            
            # Check for corresponding test file
            test_file = file_path.parent.parent / "tests" / f"test_{file_path.name}"
            analysis['has_tests'] = test_file.exists()
            
            # Rate complexity based on line count and function count
            if analysis['line_count'] > 1000 or analysis['function_count'] > 20:
                analysis['complexity_rating'] = 'high'
            elif analysis['line_count'] < 200 and analysis['function_count'] < 5:
                analysis['complexity_rating'] = 'low'
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate the complete analysis report."""
        
        print("Analyzing file structure...")
        file_structure = self.analyze_file_structure()
        
        print("Analyzing source code metrics...")
        code_metrics = self.analyze_source_code_metrics()
        
        print("Analyzing dependencies...")
        dependencies = self.analyze_dependencies()
        
        print("Analyzing physics components...")
        physics_components = self.analyze_physics_components()
        
        # Generate summary statistics
        summary = {
            'project_size': {
                'total_files': file_structure['total_files'],
                'python_files': file_structure['python_files'],
                'source_files': file_structure['source_files'],
                'total_lines': code_metrics['total_lines'],
                'total_functions': code_metrics['total_functions'],
                'total_classes': code_metrics['total_classes']
            },
            'code_quality': {
                'avg_complexity': code_metrics.get('avg_complexity', 0),
                'max_complexity': code_metrics.get('max_complexity', 0),
                'docstring_coverage': code_metrics['docstring_coverage'],
                'error_handling_coverage': code_metrics['error_handling_coverage']
            },
            'architecture': {
                'main_modules': len(code_metrics['module_analysis']),
                'external_dependencies': len(dependencies['external_packages']),
                'internal_dependencies': len(dependencies['internal_modules'])
            }
        }
        
        return {
            'timestamp': '2025-09-11T01:45:00Z',
            'summary': summary,
            'file_structure': file_structure,
            'code_metrics': code_metrics,
            'dependencies': dependencies,
            'physics_components': physics_components
        }


def main():
    """Run comprehensive codebase analysis."""
    
    analyzer = CodebaseAnalyzer('.')
    report = analyzer.generate_comprehensive_report()
    
    # Save detailed report
    with open('codebase_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nCOMPREHENSIVE CODEBASE ANALYSIS COMPLETE")
    print("=" * 50)
    
    # Print summary
    summary = report['summary']
    print(f"Project Size:")
    print(f"  Total Files: {summary['project_size']['total_files']}")
    print(f"  Python Files: {summary['project_size']['python_files']}")
    print(f"  Source Files: {summary['project_size']['source_files']}")
    print(f"  Total Lines of Code: {summary['project_size']['total_lines']:,}")
    print(f"  Functions: {summary['project_size']['total_functions']}")
    print(f"  Classes: {summary['project_size']['total_classes']}")
    
    print(f"\nCode Quality:")
    print(f"  Average Complexity: {summary['code_quality']['avg_complexity']:.2f}")
    print(f"  Max Complexity: {summary['code_quality']['max_complexity']}")
    print(f"  Docstring Coverage: {summary['code_quality']['docstring_coverage']:.1%}")
    print(f"  Error Handling Coverage: {summary['code_quality']['error_handling_coverage']:.1%}")
    
    print(f"\nArchitecture:")
    print(f"  Main Modules: {summary['architecture']['main_modules']}")
    print(f"  External Dependencies: {summary['architecture']['external_dependencies']}")
    print(f"  Internal Dependencies: {summary['architecture']['internal_dependencies']}")
    
    print(f"\nDetailed report saved to: codebase_analysis_report.json")
    
    return report


if __name__ == "__main__":
    main()