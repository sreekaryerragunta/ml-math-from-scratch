import json
import glob
import os

def fix_notebook_formatting(notebook_path):
    """Fix source code formatting in notebooks - add proper newlines."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code' and 'source' in cell:
                source = cell['source']
                
                # Check if source is a list and needs fixing
                if isinstance(source, list):
                    fixed_source = []
                    for line in source:
                        # Ensure each line ends with \n if it doesn't already
                        if line and not line.endswith('\n'):
                            fixed_source.append(line + '\n')
                        else:
                            fixed_source.append(line)
                    
                    if fixed_source != source:
                        cell['source'] = fixed_source
                        modified = True
        
        if modified:
            # Save the fixed notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=4, ensure_ascii=False)
            print(f'✓ Fixed: {notebook_path}')
            return True
        else:
            print(f'  OK: {notebook_path}')
            return False
            
    except Exception as e:
        print(f'✗ Error fixing {notebook_path}: {e}')
        return False

# Find all notebooks
notebooks = glob.glob('**/*.ipynb', recursive=True)
print(f'Found {len(notebooks)} notebooks\n')

fixed_count = 0
for nb_path in sorted(notebooks):
    if fix_notebook_formatting(nb_path):
        fixed_count += 1

print(f'\nFixed {fixed_count} notebooks')
