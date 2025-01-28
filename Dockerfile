FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy code
COPY rp_handler.py /app/rp_handler.py

# Install dependencies to call huggingface
RUN pip install --upgrade pip
RUN pip install huggingface-hub

ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Download the model into the network volume
RUN mkdir -p /mnt/model && \
    huggingface-cli login --token ${HUGGINGFACE_TOKEN} && \
    huggingface-cli snapshot-download black-forest-labs/FLUX.1-dev --cache-dir /mnt/model

# Install remaining packages
RUN pip install torch torchvision diffusers runpod transformers accelerate

# Expose the port (if required by RunPod)
EXPOSE 8000

# Set the handler as the entry point
CMD ["python3", "rp_handler.py"]
