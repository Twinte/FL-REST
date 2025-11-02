# 1. Start from an official PyTorch image with CUDA support
#    This saves us from installing PyTorch and the CUDA toolkit manually.
#    You can change 'cu121' to your specific CUDA version (e.g., cu118)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Set up the working directory inside the container
WORKDIR /app

# 3. Set the PYTHONPATH
#    This allows Python to find your 'client' and 'server' modules,
#    which is crucial for imports like 'from client.model import get_model'.
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y curl

# 4. Copy and install requirements
#    We copy *only* this file first to leverage Docker's build cache.
#    This layer won't be rebuilt unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project code into the container
COPY . .

# 6. Default command (this will be overridden by docker-compose)
CMD ["python"]