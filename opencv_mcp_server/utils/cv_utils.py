from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from .config import AUTO_OPEN_OUTPUTS, Path
from .path_utils import safe_path

logger = logging.getLogger("merlin-cv-mcp.cv_utils")

def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    if image is None:
        return {"error": "Image is None"}
    
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    dtype = str(image.dtype)
    size_bytes = image.nbytes
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2)
    }

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_and_display(img: np.ndarray, original_path: str, operation: str) -> str:
    """
    Save image to file and optionally display it.
    """
    is_video_frame = any(marker in os.path.basename(original_path) for marker in 
                         ["_frame_", "_track_", "_motion_"])
    
    base_name = os.path.basename(original_path)
    name_parts = os.path.splitext(base_name)
    timestamp = get_timestamp()
    new_filename = f"{name_parts[0]}_{operation}_{timestamp}{name_parts[1]}"
    
    directory = os.path.dirname(original_path) or '.'
    new_path = str(safe_path(os.path.join(directory, new_filename)))
    
    cv2.imwrite(new_path, img)
    
    if AUTO_OPEN_OUTPUTS:
        # We could add the opening logic here if needed, but for now we focus on saving.
        pass
        
    return new_path

def validate_int_param(name: str, value: Any, minimum: int, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter '{name}' must be an integer.")
    if parsed < minimum:
        raise ValueError(f"Parameter '{name}' must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"Parameter '{name}' must be <= {maximum}.")
    return parsed

def validate_float_param(name: str, value: Any, minimum: float, maximum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter '{name}' must be a number.")
    if parsed < minimum:
        raise ValueError(f"Parameter '{name}' must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"Parameter '{name}' must be <= {maximum}.")
    return parsed
