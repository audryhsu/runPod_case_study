FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY my_handler.py /app/my_handler.py

# Install dependencies to call huggingface
RUN pip install --upgrade pip
RUN pip install torch diffusers runpod transformers accelerate huggingface_hub

# Set environment variables
ENV MODEL_DIR=/app/model
RUN mkdir -p $MODEL_DIR

# Download the Stable Diffusion model (no authentication needed)
RUN huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir $MODEL_DIR --local-dir-use-symlinks False

# Expose the port (if required by RunPod)
EXPOSE 8000

# Set the handler as the entry point
CMD ["python3", "my_handler.py"]
