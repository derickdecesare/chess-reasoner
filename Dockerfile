FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# System dependencies - add -y to avoid prompts
RUN apt-get update && \
    apt-get install -y stockfish && \
    apt-get clean

# By default, /usr/games is not on PATH. So add it:
ENV PATH="/usr/games:${PATH}"

# Now "which stockfish" should succeed
RUN which stockfish

# Install dependencies
RUN pip install transformers
RUN pip install chess
RUN pip install accelerate
RUN pip install tqdm
RUN pip install wandb
RUN pip install stockfish
RUN pip install psutil
RUN pip install pandas
RUN pip install pyarrow
