import os
import shutil

# Paths
image_folder = "/Users/prabhavsingh/Documents/CLASSES/Fall2025/NAVAID/MILESTONE1/GUIDANCE_METRICS/data/Images"           # folder with images
annotation_folder = "/Users/prabhavsingh/Downloads/CCnador7kotk6j95i6kN7nRuYZRe3lZ3kjzq14aqXJMW91z7dw2r8UFBJY4PuYSQcHbYvDvfsNFBwdE5JBWR8ClTrm8MogbUAoCR5sqvzPLxz4eMjtOSmJhQjQvV/train/ann"  # folder with .json annotation files
output_folder = '/Users/prabhavsingh/Documents/CLASSES/Fall2025/NAVAID/MILESTONE1/GUIDANCE_METRICS/data/Annotations'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all image files
for img_file in os.listdir(image_folder):
    if img_file.endswith(".png"):
        base_name = os.path.splitext(img_file)[0]
        json_name = base_name + ".png.json"
        json_path = os.path.join(annotation_folder, json_name)

        # Check if annotation exists
        if os.path.exists(json_path):
            # Copy to output folder
            shutil.copy(json_path, os.path.join(output_folder, json_name))
            print(f"Copied {json_name}")
        else:
            print(f"Annotation not found for {img_file}")
