# Build image

docker build --platform linux/amd64 -t chess-reasoner .

# Basic container access (no file sync)

docker run --rm -it chess-reasoner bash

# Development container (with file sync)

docker run --rm -it -v $(pwd)/src:/workspace/src chess-reasoner bash

# with more memory 16g of ram

docker run --rm -it --memory=16g -v $(pwd)/src:/workspace/src chess-reasoner bash

# and creates volume for model cache so it persists between restarts

docker run --rm -it --memory=16g --memory-swap=16g -v $(pwd)/src:/workspace/src chess-reasoner bash

# forces zero swap

docker run --rm -it \
 --memory=20g \
 --memory-swap=20g \
 -v $(pwd)/src:/workspace/src \
 chess-reasoner bash

# Development container with both src and models volumes

docker run --rm -it \
 --memory=20g \
 --memory-swap=20g \
 -v $(pwd)/src:/workspace/src \
 -v $(pwd)/models:/workspace/models \
 chess-reasoner bash

# Dev workflow:

1. Start dev container with file sync (command above)
2. Edit files in ./src using local IDE
3. Test in container's CLI: python /workspace/src/your_file.py
4. Changes sync automatically between host and container
5. Container is fresh on restart, but src files persist

# Flags explained:

--rm : Remove container after exit
-it : Interactive terminal
-v : Mount volume (sync files)
$(pwd)/src : Local src directory
/workspace/src: Container src directory

# Exit container

exit

####################################

# Notes Runpod deployment:

1. Choose pod
   -- A100 PCIe ($1.64 per hour)
   -- 80gb vram --> perfect for 3 bill param model testing

2. Deploy
   -- Upload docker image...
   -- mount volumes for code and models
   -- enable global networking?

# First, push your image to Docker Hub

docker build -t yourusername/chess-reasoner .
docker push yourusername/chess-reasoner

# On runpod

Enter docker image name
configure volume mounts in the UI
Start the pod

3.Access
-- runppod provides terminal access
-- can view logs through dashboard

4. Cost managment
   -- stop pod when not in use

# First need to test docker image locally before pushing it to the hub

# Build the image

docker build -t chess-reasoner .

# Run container with volumes mounted

docker run --rm -it \
 --memory=20g \
 --memory-swap=20g \
 -v $(pwd)/src:/workspace/src \
 -v $(pwd)/models:/workspace/models \
 chess-reasoner bash

# Inside container, test each script:

python src/pgn_finetuning.py
python src/cot_finetuning.py
python src/rl_training_loop_trl.py
