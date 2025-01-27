import runpod
import torch
import io
import base64
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model = pipeline("text-to-image", model="black-forest-labs/FLUX.1-dev", device=0 if device == "cuda" else -1)

def handler(event):
    input = event['input']
    prompt = input.get('prompt')
    print("Prompt = ", prompt)

    # Placeholder for a task; replace with image or text generation logic as needed
    image = model(prompt).images[0]

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