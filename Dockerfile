FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG BASE_URL

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /workspace

# Copy the model into the container first
# This layer will be cached unless the model changes
COPY llama3.2 /workspace/llama3.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Copy the src directory containing app.py
# This is likely to change more often, so we put it near the end
COPY src ./src

# Set the command to run your script
CMD ["python3", "src/app.py"]
