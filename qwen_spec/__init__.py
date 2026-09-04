"""Tree speculative decoding for Qwen3.5/3.6/3.8 hybrid models on Apple Silicon (MLX)."""
from .engine import DEFAULT_ACCEPT, MODELS, Engine

__version__ = "0.1.0"
__all__ = ["Engine", "MODELS", "DEFAULT_ACCEPT"]
