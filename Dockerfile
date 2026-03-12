FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# No system-level dependencies needed.
# Stockfish is NOT required — the training loop uses precomputed evals from a parquet file.

RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    datasets>=2.14.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    torch>=2.1.0 \
    wandb \
    pandas \
    pyarrow \
    trl \
    chess \
    tqdm \
    peft \
    einops \
    sentencepiece \
    flash-attn --no-build-isolation \
    vllm


# Set working directory
WORKDIR /workspace


# Default command to open bash
CMD ["/bin/bash"]