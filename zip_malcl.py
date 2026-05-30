import zipfile
import os

zip_name = "c:/FederatedLearning/MalCL_code_kaggle.zip"
root_dir = "c:/FederatedLearning/MalCL/MalCL_torch"

exclude_dirs = {'__pycache__', 'logs', 'results', '.ipynb_checkpoints'}
# Keep only .py files and potentially requirements.txt
include_exts = {'.py'}

print(f"Creating zip: {zip_name}")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    count = 0
    for root, dirs, files in os.walk(root_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in include_exts or file == 'requirements.txt':
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, arcname)
                print(f"  Adding: {arcname}")
                count += 1

print(f"Done! Added {count} files.")
