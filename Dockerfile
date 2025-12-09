# 1. Use a standard Python base image (Debian-based is reliable)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# 2. Set working directory
WORKDIR /app

# 3. Set PYTHONPATH so python can find your modules
ENV PYTHONPATH=/app

# 4. Install system tools (curl/git are often needed)
RUN apt-get update && apt-get install -y curl build-essential

# 5. Install PyTorch 2.9.1 with CUDA 13.0 (RTX 50 Support)
#    We run this FIRST to ensure it's the foundation of our environment.
#    (This matches the command from your screenshot)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 6. Copy and install the rest of the dependencies
#    (Make sure you removed 'torch' from this file in Step 1!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application
COPY . .

# 8. Default command
CMD ["python"]