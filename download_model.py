from huggingface_hub import snapshot_download
from huggingface_hub import login
import os

login(token=os.getenv("HUGGINGFACE_TOKEN"))
# Path to download the model in the mounted volume
target_path = "/mnt/model/FLUX.1-dev"

# Download the model
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    cache_dir=target_path,
    use_auth_token=True
)
print(f"Model downloaded to {target_path}")