import os
import glob

def remove_bom(file_path):
    """Remove UTF-8 BOM from file if present."""
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Check for BOM and remove it
    if content.startswith(b'\xef\xbb\xbf'):
        print(f'Removing BOM from: {file_path}')
        content = content[3:]  # Remove BOM
        with open(file_path, 'wb') as f:
            f.write(content)
        return True
    return False

# Find all .ipynb files
notebooks = glob.glob('**/*.ipynb', recursive=True)

fixed_count = 0
for notebook in notebooks:
    if remove_bom(notebook):
        fixed_count += 1

print(f'\nFixed {fixed_count} notebooks')
print(f'Total notebooks: {len(notebooks)}')
