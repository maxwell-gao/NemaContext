"""
Entry point for running the downloader as a module.

Usage:
    python -m src.data.downloader --source all
    python -m src.data.downloader --source packer --data-dir custom/path
"""

from .main import main

if __name__ == "__main__":
    main()
