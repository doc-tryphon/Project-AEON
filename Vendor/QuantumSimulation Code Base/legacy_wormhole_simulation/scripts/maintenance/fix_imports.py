#!/usr/bin/env python3
"""
Script to fix all relative imports by converting them to absolute imports.
This ensures the codebase can be imported properly from anywhere.
"""

import os
import re
import glob
from typing import List, Tuple

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the src directory."""
    pattern = os.path.join(directory, "src", "**", "*.py")
    return glob.glob(pattern, recursive=True)

def fix_relative_imports(file_path: str) -> bool:
    """Fix relative imports in a single file."""
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match relative imports
        patterns = [
            # from ..module import something
            (r'from \.\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', r'from src.\1'),
            # from .module import something  
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', r'from src.\1'),
            # Handle cases like from ..physics.constants import X
            (r'from \.\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', r'from src.\1'),
        ]
        
        # Apply pattern replacements
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Handle special cases where we need to determine the correct src path
        # This fixes cases where imports are nested within subdirectories
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if 'from src.' in line and file_path:
                # Extract the relative path from src to determine correct absolute path
                rel_path = os.path.relpath(file_path, os.path.join(os.getcwd(), 'src'))
                path_parts = rel_path.replace('\\', '/').split('/')
                
                # Don't modify correctly formed absolute imports
                if line.strip().startswith('from src.'):
                    # Validate that the import path exists
                    import_match = re.search(r'from src\.([a-zA-Z_][a-zA-Z0-9_\.]*)', line)
                    if import_match:
                        import_path = import_match.group(1)
                        # Convert to file path
                        expected_path = os.path.join('src', *import_path.split('.'))
                        if os.path.exists(expected_path + '.py') or os.path.exists(os.path.join(expected_path, '__init__.py')):
                            # Valid import, keep as is
                            pass
                        else:
                            print(f"  Warning: Import path may not exist: {import_path}")
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Fixed imports in {file_path}")
            return True
        else:
            print(f"  - No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all imports."""
    print("Fixing relative imports in the quantum wormhole simulation codebase...")
    
    # Find all Python files in src directory
    python_files = find_python_files('.')
    
    if not python_files:
        print("No Python files found in src directory!")
        return
    
    print(f"Found {len(python_files)} Python files to process.")
    
    fixed_count = 0
    total_files = len(python_files)
    
    # Process each file
    for file_path in python_files:
        if fix_relative_imports(file_path):
            fixed_count += 1
    
    print(f"\nImport fixing completed!")
    print(f"Files processed: {total_files}")
    print(f"Files modified: {fixed_count}")
    print(f"Files unchanged: {total_files - fixed_count}")

if __name__ == "__main__":
    main()