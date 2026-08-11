FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Fully offline: albumentations otherwise attempts a version-check network
# call on import (fails gracefully under --network none, but avoid it
# entirely rather than leave any network attempt in an offline-mandated container)
ENV NO_ALBUMENTATIONS_UPDATE=1

# System dependencies (required by OpenCV / albumentations etc.)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch/torchvision intentionally NOT here -- the base
# image already has a CUDA-12.1-correct torch 2.3.1; reinstalling a pinned
# version from requirements.txt risks a CPU-only or CUDA-mismatched wheel)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vendored DINOv2 repo (facebookresearch/dinov2, local copy) -- lets
# model_factory.py call torch.hub.load(..., source="local", pretrained=False)
# with zero network access, satisfying --network none.
COPY dinov2_repo /app/dinov2_repo

# Model code + weights
COPY model_factory.py model.py predict.py /app/
COPY best_model.pth /app/best_model.pth

ENTRYPOINT ["python3", "/app/predict.py"]
