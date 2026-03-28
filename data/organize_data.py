# organize_data_fixed.py

import os
import shutil

source = r"E:\project_NareshIT\data\raw\03_Wheat_Disease\train\RGB"
target = r"E:\project_NareshIT\data\train"

# Create folders
for cls in ["Healthy", "Rust", "Other"]:
    os.makedirs(os.path.join(target, cls), exist_ok=True)

count = 0

for file in os.listdir(source):
    filename = file.lower()

    
    if filename.endswith(".png") or filename.endswith(".tif"):

        if "health" in filename:
            label = "Healthy"
        elif "rust" in filename:
            label = "Rust"
        elif "other" in filename:
            label = "Other"
        else:
            continue

        shutil.copy(
            os.path.join(source, file),
            os.path.join(target, label, file)
        )
        count += 1

print(f"Copied {count} files")