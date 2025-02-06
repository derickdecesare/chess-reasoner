import os
import psutil


def get_docker_memory_limit():
    """
    Attempt to read the container memory limit from either cgroup v1 or v2:
    - cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    - cgroup v2: /sys/fs/cgroup/memory.max
    Returns:
        (int) memory limit in bytes, or None if no limit / file not found / "max"
    """
    path_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    path_v2 = "/sys/fs/cgroup/memory.max"

    mem_limit = None

    if os.path.isfile(path_v1):
        with open(path_v1, 'r') as f:
            val = f.read().strip()
        try:
            mem_limit = int(val)
        except ValueError:
            mem_limit = None

        # Some systems might set this to a very large number if unlimited
        if mem_limit is not None and mem_limit >= 2**63 - 1:
            mem_limit = None

    elif os.path.isfile(path_v2):
        with open(path_v2, 'r') as f:
            val = f.read().strip()
        # If val is "max" => no limit
        if val.isdigit():
            mem_limit = int(val)
            # Again, if extremely large, treat as no limit
            if mem_limit >= 2**63 - 1:
                mem_limit = None
        else:
            # "max" or unrecognized => no limit
            mem_limit = None

    return mem_limit


def print_memory_stats():
    """Print detailed memory information."""
    print("\n=== Memory Stats ===")
    
    # Docker container limit (if any)
    mem_limit = get_docker_memory_limit()
    if mem_limit is not None:
        print(f"Detected Docker memory limit: {mem_limit / (1024 ** 3):.2f} GB")
    else:
        print("No enforced Docker memory limit (or not found).")

    # System memory
    vm = psutil.virtual_memory()
    print(f"Total system memory: {vm.total / (1024**3):.2f} GB")
    print(f"Available memory: {vm.available / (1024**3):.2f} GB")
    print(f"Used memory: {vm.used / (1024**3):.2f} GB")
    print(f"Memory percent used: {vm.percent}%")





