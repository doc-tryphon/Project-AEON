import os

def count_python_files(directory):
    py_files = []
    total_lines = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                py_files.append(filepath)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    total_lines += lines
    
    return len(py_files), total_lines

def count_test_methods(directory):
    test_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if 'def test_' in line:
                            test_count += 1
    return test_count

# Run counts
src_files, src_lines = count_python_files('src')
test_files, test_lines = count_python_files('tests')
test_methods = count_test_methods('tests')

print(f"Source files: {src_files}")
print(f"Source lines: {src_lines}")
print(f"Test files: {test_files}")
print(f"Test lines: {test_lines}")
print(f"Test methods: {test_methods}")