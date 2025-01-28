import runpod
import torch
import os
import io
import base64
from diffusers import FluxPipeline
from huggingface_hub import login, snapshot_download

# first download the model in the mount directory

# Fetch token from the environment
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set.")

# Log in
login(token=huggingface_token)

# Path to download the model in the mounted volume
target_path = "/mnt/model/FLUX.1-dev"

# Download the model
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    cache_dir=target_path,
    use_auth_token=huggingface_token
)
print(f"Model downloaded to {target_path}")

MODEL_PATH = "/mnt/model-storage/FLUX.1-dev"

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