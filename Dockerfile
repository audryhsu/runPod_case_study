FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY rp_handler.py /app/rp_handler.py
COPY download_model.py /app/download_model.py

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision diffusers runpod transformers accelerate

VOLUME ["/mnt/model"]
RUN python3 /app/download_model.py

# Expose the port (if required by RunPod)
EXPOSE 8000

# Set the handler as the entry point
CMD ["python3", "rp_handler.py"]
