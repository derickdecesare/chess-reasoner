FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# System dependencies - add -y to avoid prompts
RUN apt-get update && \
    apt-get install -y stockfish && \
    apt-get clean

# By default, /usr/games is not on PATH. So add it:
ENV PATH="/usr/games:${PATH}"

# Now "which stockfish" should succeed
RUN which stockfish

# # Install dependencies
# RUN pip install transformers
# RUN pip install chess
# RUN pip install accelerate
# RUN pip install tqdm
# RUN pip install wandb
# RUN pip install stockfish
# RUN pip install psutil
# RUN pip install pandas
# RUN pip install pyarrow


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
    stockfish \
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