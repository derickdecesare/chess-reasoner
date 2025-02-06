source venv/bin/activate

this is where the model will be located and how to delete it

rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B

d# Navigate to the cache directory

cd ~/.cache/huggingface

# List contents

ls

# See size of directories

du -sh \*

caffeinate -i -s python create_pgn_dataset.py
