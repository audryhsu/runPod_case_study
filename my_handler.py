import runpod
import json
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import AutoTokenizer
from huggingface_hub import login
import torch
import os

# Initialize global variables
model_id = "stability-ai/stable-diffusion-2-1-base"  # Choose a lightweight model
cache_dir = "/tmp/model"  # Use /tmp for serverless environments

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set.")

# Load model once to reuse across invocations
pipeline = None

def load_model():
    global pipeline

    login(token=huggingface_token)

    if pipeline is None:
        print("Loading Stable Diffusion model...")
        
        # Load the Stable Diffusion model and tokenizer
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        pipeline = pipeline.to("cuda")
        
        print("Model loaded successfully.")
    else:
        print("Model already loaded.")

def handler(event, context=None):
    # Ensure the model is loaded
    load_model()

    # Parse input
    try:
        body = json.loads(event.get("input", "{}"))
        prompt = body.get("prompt", "A futuristic cityscape at sunset")
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": "Invalid JSON input."}

    if not prompt:
        return {"statusCode": 400, "body": "Prompt is required."}

    # Generate image
    try:
        print(f"Generating image for prompt: {prompt}")
        images = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images
        image_data = []

        for idx, img in enumerate(images):
            # Convert image to a format suitable for response
            img_path = f"/tmp/image_{idx}.png"
            img.save(img_path)
            with open(img_path, "rb") as image_file:
                image_data.append(image_file.read())

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/octet-stream"},
            "body": image_data[0]  # Send first image as the response
        }

    except Exception as e:
        print(f"Error generating image: {e}")
        return {"statusCode": 500, "body": str(e)}

# Debugging locally
if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
