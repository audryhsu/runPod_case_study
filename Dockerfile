FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY rp_handler.py /app/rp_handler.py

# Install dependencies to call huggingface
RUN pip install --upgrade pip

# Install remaining packages
RUN pip install torch torchvision diffusers runpod transformers accelerate

# Expose the port (if required by RunPod)
EXPOSE 8000

# Set the handler as the entry point
CMD ["python3", "rp_handler.py"]
