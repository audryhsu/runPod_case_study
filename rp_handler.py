import runpod
import torch
import os
import io
import base64
from diffusers import FluxPipeline
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_TOKEN"))
print(os.getenv("HUGGINGFACE_TOKEN"))
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_auth_token=True
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