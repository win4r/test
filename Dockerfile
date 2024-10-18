# 构建阶段
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

ARG BASE_URL

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# 最终阶段
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as v1

ARG BASE_URL

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Set the working directory in the container
WORKDIR /workspace

# Copy the model into the container
COPY llama3.2 /workspace/llama3.2

# Copy the src directory containing app.py
COPY src ./src

# Set the command to run your script
CMD ["python3", "src/app.py"]