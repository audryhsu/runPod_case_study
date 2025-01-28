import runpod
import torch
import os
import io
import base64
from diffusers import FluxPipeline
from huggingface_hub import login, snapshot_download
import shutil

# Fetch token from the environment
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set.")

# Log in
login(token=huggingface_token)

# Path to download the model in the mounted volume
target_path = "/mnt/model/FLUX.1-schnell"

if not os.path.exists(target_path):
    print("Creating the directory")
    os.makedirs(target_path)
else:
    print(f"{target_path} already present")

if not os.path.exists(target_path) or not os.listdir(target_path):
    print("Model not found. Downloading the model...")
    # Download the model
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-schnell",
        cache_dir=target_path,
        use_auth_token=huggingface_token
    )
    print(f"Model downloaded to {target_path}")
else:
    print(f"Model already exists at {target_path}. Skipping download.")

# Get disk usage statistics for /mnt
total, used, free = shutil.disk_usage("/mnt")

print(f"Total: {total // (2**30)} GB")
print(f"Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")

MODEL_PATH = "/mnt/model/FLUX.1-schnell"

# Load the pipeline from the pre-downloaded model
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")


def handler(event):
    input = event['input']
    prompt = input.get('prompt')
    print("Prompt = ", prompt)

    # Placeholder for a task; replace with image or text generation logic as needed
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0)  # Ensure deterministic generation
    ).images[0]

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "image/png"
        },
        "body": base64.b64encode(buffered.getvalue()).decode('utf-8'),
        "isBase64Encoded": True
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})