import os
import shutil
from huggingface_hub import HfApi, Repository

# Your Hugging Face model repo ID
repo_id = "JagadeeshAi/BID"

# Path to your source directory (where the `.pth` files are somewhere inside)
source_root = "./"  # You can replace this with your actual root path

# Temporary folder to build the upload tree
temp_repo_dir = "./temp_hf_upload"

# Step 1: Clone the HF repo locally (clean up old if needed)
if os.path.exists(temp_repo_dir):
    shutil.rmtree(temp_repo_dir)

repo = Repository(local_dir=temp_repo_dir, clone_from=repo_id)

# Step 2: Walk through your source directory and copy all .pth files
for dirpath, _, filenames in os.walk(source_root):
    for filename in filenames:
        if filename.endswith(".pth"):
            abs_file_path = os.path.join(dirpath, filename)
            # Preserve folder structure relative to source_root
            rel_path = os.path.relpath(abs_file_path, source_root)
            dest_path = os.path.join(temp_repo_dir, rel_path)

            # Create destination folder if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the file
            shutil.copy2(abs_file_path, dest_path)
            print(f"Copied: {rel_path}")

# Step 3: Push to Hugging Face
repo.push_to_hub(commit_message="Upload all .pth files with structure preserved")
