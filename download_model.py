from huggingface_hub import snapshot_download
import os

print("Trying to find all ENVIRONMENT VARIABLES:", os.environ)
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
# Path to download the model in the mounted volume
target_path = "/mnt/model/FLUX.1-dev"

# Download the model
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    cache_dir=target_path,
    token=token
)
print(f"Model downloaded to {target_path}")