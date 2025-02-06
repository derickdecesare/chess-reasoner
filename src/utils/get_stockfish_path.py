import os
import platform
import subprocess

def get_stockfish_path():
    """Get Stockfish path across different environments (Docker, Mac, Linux)."""
    
    # First check if path is provided in environment variable
    if 'STOCKFISH_PATH' in os.environ:
        return os.environ['STOCKFISH_PATH']
    
    # Try to get from 'which' command first
    try:
        # On Unix-like systems (Mac, Linux), this will work
        stockfish_path = subprocess.check_output(['which', 'stockfish'], 
                                               universal_newlines=True).strip()
        if os.path.exists(stockfish_path):
            return stockfish_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Common paths based on OS
    if platform.system() == 'Darwin':  # macOS
        possible_paths = [
            '/opt/homebrew/bin/stockfish',  # Homebrew on Apple Silicon
            '/usr/local/bin/stockfish',     # Homebrew on Intel Mac
        ]
    else:  # Linux (including Docker)
        possible_paths = [
            '/usr/games/stockfish',     # Default Linux/Docker install location
            '/usr/local/bin/stockfish', # Alternative Linux location
            '/usr/bin/stockfish',       # Another possible Linux location
        ]
    
    # Check all possible paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        "Stockfish not found. Please ensure it's installed and either:\n"
        "1. Available in PATH\n"
        "2. In a standard location\n"
        "3. Specified via STOCKFISH_PATH environment variable"
    )
