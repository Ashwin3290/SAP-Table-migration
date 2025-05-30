#!/usr/bin/env python3
"""
Function Usage Analyzer - Analyze function definitions and calls across Python files
Uses AST to identify unused functions and create dependency graphs
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class FunctionInfo:
    """Information about a function definition"""
    name: str
    filename: str
    lineno: int
    is_method: bool = False
    class_name: Optional[str] = None
    is_private: bool = False
    is_dunder: bool = False
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

@dataclass
class CallInfo:
    """Information about a function call"""
    name: str
    filename: str
    lineno: int
    context: str  # The line of code where the call happens
    is_method_call: bool = False
    object_name: Optional[str] = None

class FunctionUsageAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze function definitions and calls"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.functions: List[FunctionInfo] = []
        self.calls: List[CallInfo] = []
        self.imports: Dict[str, str] = {}  # alias -> module
        self.from_imports: Dict[str, str] = {}  # name -> module
        self.current_class = None
        self.source_lines = []
        
    def analyze_file(self, filepath: str) -> Tuple[List[FunctionInfo], List[CallInfo], Dict[str, str]]:
        """Analyze a Python file and return functions, calls, and imports"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.source_lines = content.splitlines()
            
            tree = ast.parse(content, filename=filepath)
            self.visit(tree)
            
            return self.functions, self.calls, {**self.imports, **self.from_imports}
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return [], [], {}
    
    def visit_Import(self, node):
        """Handle import statements"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Handle from...import statements"""
        if node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.from_imports[name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Handle class definitions"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Handle function definitions"""
        self._visit_function_def(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Handle async function definitions"""
        self._visit_function_def(node)
    
    def _visit_function_def(self, node):
        """Common logic for function definitions"""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{ast.unparse(decorator.value)}.{decorator.attr}")
        
        # Get docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        func_info = FunctionInfo(
            name=node.name,
            filename=self.filename,
            lineno=node.lineno,
            is_method=self.current_class is not None,
            class_name=self.current_class,
            is_private=node.name.startswith('_') and not node.name.startswith('__'),
            is_dunder=node.name.startswith('__') and node.name.endswith('__'),
            docstring=docstring,
            decorators=decorators
        )
        
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Handle function calls"""
        call_name = None
        is_method_call = False
        object_name = None
        
        if isinstance(node.func, ast.Name):
            # Simple function call: func()
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method() or module.func()
            call_name = node.func.attr
            is_method_call = True
            if isinstance(node.func.value, ast.Name):
                object_name = node.func.value.id
            else:
                object_name = ast.unparse(node.func.value)
        
        if call_name:
            # Get the source line for context
            context = ""
            if hasattr(node, 'lineno') and node.lineno <= len(self.source_lines):
                context = self.source_lines[node.lineno - 1].strip()
            
            call_info = CallInfo(
                name=call_name,
                filename=self.filename,
                lineno=node.lineno,
                context=context,
                is_method_call=is_method_call,
                object_name=object_name
            )
            
            self.calls.append(call_info)
        
        self.generic_visit(node)

class ProjectAnalyzer:
    """Analyze function usage across multiple Python files"""
    
    def __init__(self, project_paths: List[str]):
        self.project_paths = project_paths
        self.all_functions: Dict[str, List[FunctionInfo]] = {}
        self.all_calls: Dict[str, List[CallInfo]] = {}
        self.all_imports: Dict[str, Dict[str, str]] = {}
        
    def analyze_project(self) -> Dict:
        """Analyze all files in the project"""
        print("🔍 Analyzing project files...")
        
        for filepath in self.project_paths:
            if not os.path.exists(filepath):
                print(f"⚠️  File not found: {filepath}")
                continue
                
            print(f"📄 Analyzing: {os.path.basename(filepath)}")
            
            analyzer = FunctionUsageAnalyzer(filepath)
            functions, calls, imports = analyzer.analyze_file(filepath)
            
            self.all_functions[filepath] = functions
            self.all_calls[filepath] = calls
            self.all_imports[filepath] = imports
            
            print(f"   Found {len(functions)} functions and {len(calls)} calls")
        
        return self._generate_analysis_report()
    
    def _generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        # Collect all function names and their definitions
        defined_functions = {}
        for filepath, functions in self.all_functions.items():
            for func in functions:
                key = f"{func.name}"
                if func.is_method and func.class_name:
                    key = f"{func.class_name}.{func.name}"
                
                if key not in defined_functions:
                    defined_functions[key] = []
                defined_functions[key].append(func)
        
        # Collect all function calls
        called_functions = set()
        call_details = defaultdict(list)
        
        for filepath, calls in self.all_calls.items():
            for call in calls:
                called_functions.add(call.name)
                call_details[call.name].append(call)
                
                # Also check for method calls
                if call.is_method_call and call.object_name:
                    method_key = f"{call.object_name}.{call.name}"
                    called_functions.add(method_key)
                    call_details[method_key].append(call)
        
        # Find unused functions
        unused_functions = []
        used_functions = []
        
        for func_key, func_list in defined_functions.items():
            is_used = False
            
            # Check direct calls
            if func_key in called_functions:
                is_used = True
            
            # Check if it's a simple function name that might be called
            for func_info in func_list:
                if func_info.name in called_functions:
                    is_used = True
                    break
                
                # Special cases - don't mark these as unused
                if (func_info.name == 'main' or 
                    func_info.is_dunder or 
                    any(dec in ['property', 'staticmethod', 'classmethod'] for dec in func_info.decorators)):
                    is_used = True
                    break
            
            if is_used:
                used_functions.extend(func_list)
            else:
                unused_functions.extend(func_list)
        
        # Generate statistics
        total_functions = sum(len(funcs) for funcs in self.all_functions.values())
        total_calls = sum(len(calls) for calls in self.all_calls.values())
        
        return {
            'summary': {
                'total_files': len(self.project_paths),
                'total_functions': total_functions,
                'total_calls': total_calls,
                'used_functions': len(used_functions),
                'unused_functions': len(unused_functions),
                'usage_percentage': round((len(used_functions) / total_functions * 100), 2) if total_functions > 0 else 0
            },
            'unused_functions': unused_functions,
            'used_functions': used_functions,
            'call_details': dict(call_details),
            'function_definitions': defined_functions,
            'file_analysis': self._generate_file_analysis()
        }
    
    def _generate_file_analysis(self) -> Dict:
        """Generate per-file analysis"""
        file_analysis = {}
        
        for filepath in self.project_paths:
            if filepath not in self.all_functions:
                continue
                
            functions = self.all_functions[filepath]
            calls = self.all_calls[filepath]
            
            file_analysis[os.path.basename(filepath)] = {
                'path': filepath,
                'functions_defined': len(functions),
                'function_calls_made': len(calls),
                'functions': [
                    {
                        'name': f.name,
                        'line': f.lineno,
                        'is_method': f.is_method,
                        'class_name': f.class_name,
                        'is_private': f.is_private,
                        'decorators': f.decorators
                    } for f in functions
                ],
                'calls': [
                    {
                        'name': c.name,
                        'line': c.lineno,
                        'context': c.context,
                        'is_method_call': c.is_method_call,
                        'object_name': c.object_name
                    } for c in calls
                ]
            }
        
        return file_analysis

def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report"""
    summary = analysis['summary']
    unused = analysis['unused_functions']
    
    print("\n" + "="*80)
    print("🎯 FUNCTION USAGE ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Files analyzed: {summary['total_files']}")
    print(f"   Total functions: {summary['total_functions']}")
    print(f"   Total function calls: {summary['total_calls']}")
    print(f"   Used functions: {summary['used_functions']}")
    print(f"   Unused functions: {summary['unused_functions']}")
    print(f"   Usage rate: {summary['usage_percentage']}%")
    
    if unused:
        print(f"\n🚫 UNUSED FUNCTIONS ({len(unused)}):")
        print("-" * 60)
        
        # Group by file
        unused_by_file = defaultdict(list)
        for func in unused:
            unused_by_file[os.path.basename(func.filename)].append(func)
        
        for filename, funcs in unused_by_file.items():
            print(f"\n📄 {filename}:")
            for func in funcs:
                func_type = "method" if func.is_method else "function"
                class_info = f" (in {func.class_name})" if func.class_name else ""
                privacy = " [private]" if func.is_private else ""
                decorators = f" @{', @'.join(func.decorators)}" if func.decorators else ""
                
                print(f"   Line {func.lineno:3d}: {func_type} {func.name}{class_info}{privacy}{decorators}")
    else:
        print(f"\n✅ Great! All functions are being used.")
    
    print(f"\n📋 FILE BREAKDOWN:")
    print("-" * 60)
    for filename, info in analysis['file_analysis'].items():
        print(f"📄 {filename}:")
        print(f"   Functions defined: {info['functions_defined']}")
        print(f"   Function calls made: {info['function_calls_made']}")

def main():
    """Main function to run the analyzer"""
    # Your project files - update these paths
    project_files = [
        "planner.py",
        "executor.py", 
        "generator.py",
        "dmtool.py",
        "query_analyzer.py"
    ]
    
    print("🚀 Starting Function Usage Analysis")
    print("-" * 50)
    
    # Verify files exist
    existing_files = []
    for filepath in project_files:
        if os.path.exists(filepath):
            existing_files.append(filepath)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    if not existing_files:
        print("❌ No valid Python files found!")
        return
    
    # Analyze the project
    analyzer = ProjectAnalyzer(existing_files)
    analysis = analyzer.analyze_project()
    
    # Print the report
    print_analysis_report(analysis)
    
    # Optionally save detailed report to JSON
    save_report = input("\n💾 Save detailed report to JSON? (y/n): ").lower().strip()
    if save_report == 'y':
        output_file = "function_usage_analysis.json"
        
        # Convert objects to dictionaries for JSON serialization
        json_data = {
            'summary': analysis['summary'],
            'unused_functions': [
                {
                    'name': f.name,
                    'filename': os.path.basename(f.filename),
                    'lineno': f.lineno,
                    'is_method': f.is_method,
                    'class_name': f.class_name,
                    'is_private': f.is_private,
                    'decorators': f.decorators,
                    'docstring': f.docstring[:100] + '...' if f.docstring and len(f.docstring) > 100 else f.docstring
                } for f in analysis['unused_functions']
            ],
            'file_analysis': analysis['file_analysis']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detailed report saved to: {output_file}")

if __name__ == "__main__":
    main()