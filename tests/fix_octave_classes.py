import os
import re
import sys

# Default to polarization-curve-modeling
target_dir = '/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications/polarization-curve-modeling'

if len(sys.argv) > 1:
    target_dir = sys.argv[1]

print(f"Cleaning .m files in {target_dir}")

def clean_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'rb') as f:
        content_bytes = f.read()
    
    # normalize line endings
    content = content_bytes.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    
    lines = content.splitlines()
    new_lines = []
    
    in_properties = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('properties'):
            in_properties = True
            new_lines.append(line)
            continue
            
        if in_properties:
             if stripped.startswith('methods'):
                 in_properties = False
             if stripped == 'end':
                 in_properties = False
             
        if in_properties:
            # Skip empty lines or comments
            if not stripped or stripped.startswith('%'):
                new_lines.append(line)
                continue
                
            if stripped == 'end':
                in_properties = False
                new_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) >= 2:
                prop_name = parts[0]
                second = parts[1]
                
                if second != '=':
                    # Likely a type. specific logic:
                    # Remove the type.
                    # Prop Type ... -> Prop ...
                    
                    rest = parts[2:]
                    
                    new_line = "        " + prop_name
                    if rest:
                        new_line += " " + " ".join(rest)
                    
                    new_lines.append(new_line)
                    continue

        new_lines.append(line)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')

def recursive_clean(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.m'):
                clean_file(os.path.join(root, file))

if __name__ == "__main__":
    recursive_clean(target_dir)
