FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install open_clip with training dependencies
COPY open_clip/ /workspace/open_clip/
RUN pip install --no-cache-dir /workspace/open_clip[training]

# Install MLflow and ONNX for the pipeline
RUN pip install --no-cache-dir \
    mlflow \
    onnx \
    onnxruntime

# Everything else is mounted at runtime:
#   docker run --gpus all \
#     -v /home/sumit/sumit-cicd-data-openclip-val:/workspace \
#     openclip-cicd bash pipeline.sh

ENTRYPOINT ["bash"]
