from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

def env_bool(var_name: str, default: bool = False) -> bool:
    val = os.getenv(var_name, "").lower().strip()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default

def env_int(var_name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        val = int(os.getenv(var_name, str(default)))
        if minimum is not None:
            return max(minimum, val)
        return val
    except (ValueError, TypeError):
        return default

def env_float(var_name: str, default: float, minimum: Optional[float] = None) -> float:
    try:
        val = float(os.getenv(var_name, str(default)))
        if minimum is not None:
            return max(minimum, val)
        return val
    except (ValueError, TypeError):
        return default

# Server Meta
SERVER_NAME = "merlin-cv-mcp"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# Paths & Workspace
# Agnostic support for Home and Workspace
NANOBOT_WORKSPACE = os.getenv("NANOBOT_WORKSPACE", "~/.nanobot/workspace")
SHARED_WORKSPACE_BASE = Path(os.getenv("MERLIN_CV_WORKSPACE", "." if os.getenv("NANOBOT_WORKSPACE") else "./workspace")).resolve()

# We allow multiple roots by default
ALLOWED_ROOTS = [
    Path.home().resolve(),
    Path(os.getcwd()).resolve(),
    Path(NANOBOT_WORKSPACE).expanduser().resolve()
]

# Limits & Constraints
MAX_IMAGE_DIMENSION = env_int("MERLIN_CV_MAX_IMAGE_DIMENSION", 4096, minimum=64)
MAX_VIDEO_FRAMES = env_int("MERLIN_CV_MAX_VIDEO_FRAMES", 1200, minimum=1)
MAX_CAMERA_DURATION_SECONDS = env_int("MERLIN_CV_MAX_CAMERA_DURATION_SECONDS", 60, minimum=1)
MAX_VIDEO_FPS = env_float("MERLIN_CV_MAX_VIDEO_FPS", 60.0, minimum=1.0)

# Helper functions for tool modules
def get_max_image_dimension() -> int:
    return MAX_IMAGE_DIMENSION

def get_max_video_frames() -> int:
    return MAX_VIDEO_FRAMES

def get_max_camera_duration_seconds() -> int:
    return MAX_CAMERA_DURATION_SECONDS

def get_max_video_fps() -> float:
    return MAX_VIDEO_FPS

# Hardware & UI
AUTO_OPEN_OUTPUTS = env_bool("MERLIN_CV_AUTO_OPEN", False)
CAMERA_PREVIEW = env_bool("MERLIN_CV_CAMERA_PREVIEW", False)
ENABLE_CAMERA = env_bool("MERLIN_CV_ENABLE_CAMERA", False)

# Models
MODELS_DIR_ENV = os.getenv("OPENCV_DNN_MODELS_DIR")
DEFAULT_MODELS_DIR = (Path(__file__).resolve().parent.parent.parent / "models").resolve()

def get_models_dir() -> Path:
    if MODELS_DIR_ENV:
        return Path(MODELS_DIR_ENV).expanduser().resolve()
    return DEFAULT_MODELS_DIR

# Security
ALLOW_INSECURE_URL = env_bool("MERLIN_CV_ALLOW_INSECURE_PROVIDER_URL", False)
ALLOW_PRIVATE_URL = env_bool("MERLIN_CV_ALLOW_PRIVATE_PROVIDER_URL", False)
STRICT_PATH_VALIDATION = env_bool("MERLIN_CV_STRICT_PATH_VALIDATION", True)
