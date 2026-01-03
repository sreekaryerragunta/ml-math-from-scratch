import os
import glob

# Replacement mappings
replacements = {
    'Interview Tip': 'Key Point',
    'Interview Question': 'Key Point',
    'Interview Readiness': 'Key Takeaways',
    'Interview Answer': 'Answer',
    'Interview Questions': 'Key Questions',
    'interview': 'key concept'
}

# Find all .md and .ipynb files
files = glob.glob('**/*.md', recursive=True) + glob.glob('**/*.ipynb', recursive=True)

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply all replacements
        modified = content
        for old, new in replacements.items():
            modified = modified.replace(old, new)
        
        # Only write if changed
        if modified != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f'Updated: {filepath}')
    
    except Exception as e:
        print(f'Error processing {filepath}: {e}')

print('\\nReplacement complete!')
