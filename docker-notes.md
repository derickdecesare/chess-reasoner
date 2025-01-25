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
