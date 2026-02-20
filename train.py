"""
Root-level entry point — forwards to src.train.

Prefer:  python -m src.train --config ...
Also OK: python train.py --config ...
"""
from src.train import main

if __name__ == "__main__":
    main()
