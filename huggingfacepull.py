import os
from huggingface_hub import HfApi, hf_hub_download

# Settings
repo_id = "JagadeeshAi/BID"
local_save_root = "./downloaded_models"  # Change this to wherever you want

# Step 1: Get all files in the repo
api = HfApi()
all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")

# Step 2: Filter for .pth files and download them preserving structure
for file_path in all_files:
    if file_path.endswith(".pth"):
        print(f"Downloading: {file_path}")

        # Create target folder if needed
        local_file_path = os.path.join(local_save_root, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download and write to the proper location
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="model"
        )

        # Copy to correct location (hf_hub_download saves to cache dir)
        with open(downloaded_path, "rb") as src_file:
            with open(local_file_path, "wb") as dest_file:
                dest_file.write(src_file.read())

print("✅ All .pth files downloaded and placed correctly.")
