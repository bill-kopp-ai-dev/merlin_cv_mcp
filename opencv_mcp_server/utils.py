import cv2
import numpy as np
import os
import logging
import datetime
import subprocess
import platform
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .security import record_security_event

logger = logging.getLogger("opencv-mcp-server.utils")

# Define o Workspace Compartilhado do nanobot
# Fallback para uma pasta local 'workspace' se a variável não for fornecida no config.json
WORKSPACE_ENV = os.getenv("NANOBOT_WORKSPACE", "./workspace")

# Resolvemos o caminho absoluto para evitar ambiguidades
SHARED_WORKSPACE = Path(WORKSPACE_ENV).resolve()

# Garante que a pasta exista antes de o servidor começar a operar
SHARED_WORKSPACE.mkdir(parents=True, exist_ok=True)

logger.info(f"🧙‍♂️ Merlin CV rodando. Workspace seguro travado em: {SHARED_WORKSPACE}")

AUTO_OPEN_ENV = "MERLIN_CV_AUTO_OPEN"
CAMERA_PREVIEW_ENV = "MERLIN_CV_CAMERA_PREVIEW"
CAMERA_ENABLE_ENV = "MERLIN_CV_ENABLE_CAMERA"
MAX_IMAGE_DIMENSION_ENV = "MERLIN_CV_MAX_IMAGE_DIMENSION"
MAX_VIDEO_FRAMES_ENV = "MERLIN_CV_MAX_VIDEO_FRAMES"
MAX_CAMERA_DURATION_ENV = "MERLIN_CV_MAX_CAMERA_DURATION_SECONDS"
MAX_VIDEO_FPS_ENV = "MERLIN_CV_MAX_VIDEO_FPS"
DEFAULT_MODELS_DIR = (Path(__file__).resolve().parent.parent / "models").resolve()
DEFAULT_MAX_IMAGE_DIMENSION = 4096
DEFAULT_MAX_VIDEO_FRAMES = 1200
DEFAULT_MAX_CAMERA_DURATION_SECONDS = 60
DEFAULT_MAX_VIDEO_FPS = 60.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        record_security_event(
            "invalid_env_int",
            name=name,
            value=raw,
            default=default,
        )
        logger.warning("Invalid integer for %s=%r. Using default=%s", name, raw, default)
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        record_security_event(
            "invalid_env_float",
            name=name,
            value=raw,
            default=default,
        )
        logger.warning("Invalid float for %s=%r. Using default=%s", name, raw, default)
        return default
    return max(minimum, value)


def should_auto_open_outputs() -> bool:
    """
    Whether image/video outputs should be opened with system viewers automatically.
    Disabled by default for headless and agent-driven environments.
    """
    return _env_flag(AUTO_OPEN_ENV, default=False)


def should_show_camera_preview() -> bool:
    """
    Whether camera-based tools may display OpenCV GUI preview windows.
    Disabled by default for headless and agent-driven environments.
    """
    return _env_flag(CAMERA_PREVIEW_ENV, default=False)


def should_enable_camera_access() -> bool:
    """
    Whether camera capture tools are enabled.
    Disabled by default so hardware access requires explicit opt-in.
    """
    return _env_flag(CAMERA_ENABLE_ENV, default=False)


def get_max_image_dimension() -> int:
    return _env_int(MAX_IMAGE_DIMENSION_ENV, DEFAULT_MAX_IMAGE_DIMENSION, minimum=64)


def get_max_video_frames() -> int:
    return _env_int(MAX_VIDEO_FRAMES_ENV, DEFAULT_MAX_VIDEO_FRAMES, minimum=1)


def get_max_camera_duration_seconds() -> int:
    return _env_int(
        MAX_CAMERA_DURATION_ENV,
        DEFAULT_MAX_CAMERA_DURATION_SECONDS,
        minimum=1,
    )


def get_max_video_fps() -> float:
    return _env_float(MAX_VIDEO_FPS_ENV, DEFAULT_MAX_VIDEO_FPS, minimum=1.0)


def get_models_dir() -> Path:
    """
    Resolve the directory containing DNN model assets.
    Priority:
    1) OPENCV_DNN_MODELS_DIR env var
    2) repository-local default models directory
    """
    configured = os.getenv("OPENCV_DNN_MODELS_DIR", "").strip()
    if configured:
        model_dir = Path(configured).expanduser()
        if not model_dir.is_absolute():
            model_dir = (SHARED_WORKSPACE / model_dir).resolve()
    else:
        model_dir = DEFAULT_MODELS_DIR

    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def resolve_model_asset_path(requested_path: Optional[str], default_filename: str) -> str:
    """
    Resolve model-related paths and enforce they remain inside OPENCV_DNN_MODELS_DIR.
    Relative paths are resolved against the models directory.
    """
    model_dir = get_models_dir().resolve()
    raw_path = (requested_path or "").strip()

    if not raw_path:
        target_path = (model_dir / default_filename).resolve()
    else:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            target_path = candidate.resolve()
        else:
            target_path = (model_dir / candidate).resolve()

    if not target_path.is_relative_to(model_dir):
        record_security_event(
            "model_asset_path_blocked",
            requested=requested_path,
            resolved=str(target_path),
            model_dir=str(model_dir),
        )
        msg = (
            "Model path must stay inside OPENCV_DNN_MODELS_DIR "
            f"({model_dir}). Received: {target_path}"
        )
        logger.warning(msg)
        raise PermissionError(msg)

    return str(target_path)


_CLASS_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9 _./:+()-]+")
_MAX_CLASS_LABEL_LEN = 64


def sanitize_class_label(raw_label: str, fallback: str = "unknown") -> str:
    """
    Sanitize model class labels before exposing them in tool output/prompts.
    """
    original = "" if raw_label is None else str(raw_label)
    text = original
    text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
    text = _CLASS_LABEL_SAFE_RE.sub(" ", text)
    text = " ".join(text.split())

    if len(text) > _MAX_CLASS_LABEL_LEN:
        text = text[:_MAX_CLASS_LABEL_LEN].rstrip()

    sanitized = text or fallback
    if sanitized != original.strip():
        record_security_event("class_label_sanitized")
    return sanitized


def validate_int_param(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: Optional[int] = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        record_security_event(
            "invalid_parameter_int",
            name=name,
            value=value,
            reason="not_integer",
        )
        raise ValueError(f"Parameter '{name}' must be an integer.")

    if parsed < minimum:
        record_security_event(
            "invalid_parameter_int",
            name=name,
            value=parsed,
            minimum=minimum,
            reason="below_minimum",
        )
        raise ValueError(f"Parameter '{name}' must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        record_security_event(
            "invalid_parameter_int",
            name=name,
            value=parsed,
            maximum=maximum,
            reason="above_maximum",
        )
        raise ValueError(f"Parameter '{name}' must be <= {maximum}.")
    return parsed


def validate_float_param(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        record_security_event(
            "invalid_parameter_float",
            name=name,
            value=value,
            reason="not_number",
        )
        raise ValueError(f"Parameter '{name}' must be a number.")

    if parsed < minimum:
        record_security_event(
            "invalid_parameter_float",
            name=name,
            value=parsed,
            minimum=minimum,
            reason="below_minimum",
        )
        raise ValueError(f"Parameter '{name}' must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        record_security_event(
            "invalid_parameter_float",
            name=name,
            value=parsed,
            maximum=maximum,
            reason="above_maximum",
        )
        raise ValueError(f"Parameter '{name}' must be <= {maximum}.")
    return parsed

def safe_path(requested_path: str) -> Path:
    """
    Garante que o caminho solicitado está estritamente dentro do SHARED_WORKSPACE.
    Converte caminhos relativos para absolutos usando o workspace como base.
    """
    # 1. Cria o objeto Path
    target_path = Path(requested_path)
    
    # 2. Se o LLM mandou um caminho relativo (ex: "imagem1.png"), 
    # nós o anexamos ao workspace base.
    if not target_path.is_absolute():
        target_path = SHARED_WORKSPACE / target_path
        
    # 3. Resolve atalhos e navegações como '..' ou symlinks
    try:
        target_path = target_path.resolve()
    except Exception as e:
        record_security_event(
            "workspace_path_invalid",
            requested=requested_path,
        )
        raise ValueError(f"Caminho inválido ou inacessível: {requested_path}")

    # 4. A TRAVA DE SEGURANÇA (Obrigatório Python >= 3.9)
    # Verifica matematicamente se o caminho resolvido é filho do workspace
    if not target_path.is_relative_to(SHARED_WORKSPACE):
        record_security_event(
            "workspace_path_blocked",
            requested=requested_path,
            resolved=str(target_path),
            workspace=str(SHARED_WORKSPACE),
        )
        msg = f"Acesso negado: Tentativa de violação de limite de diretório ({target_path})"
        logger.warning(msg)
        raise PermissionError(msg)
        
    return target_path

# Utility Functions
def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about an image
    
    Args:
        image: OpenCV image
        
    Returns:
        Dict: Image information including dimensions, channels, etc.
    """
    if image is None:
        raise ValueError("Image is None")
    
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
    """
    Get current timestamp as a string
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def open_image_with_system_viewer(image_path: str) -> None:
    """
    Open an image with the system's default image viewer
    
    Args:
        image_path: Path to the image file
    """
    if not should_auto_open_outputs():
        return

    image_path = str(safe_path(image_path))

    # Platform-specific image opening commands
    system = platform.system()
    
    try:
        if system == 'Windows':
            os.startfile(image_path)
        elif system == 'Darwin':  # macOS
            subprocess.call(['open', image_path])
        else:  # Linux and other Unix-like systems
            subprocess.call(['xdg-open', image_path])
        
        logger.info(f"Opened image: {image_path}")
    except Exception as e:
        logger.error(f"Error opening image with system viewer: {e}")
        # Continue execution even if display fails

def open_video_with_system_viewer(video_path: str) -> None:
    """
    Open a video with the system's default video player
    
    Args:
        video_path: Path to the video file
    """
    if not should_auto_open_outputs():
        return

    video_path = str(safe_path(video_path))

    # Platform-specific video opening commands
    system = platform.system()
    
    try:
        if system == 'Windows':
            os.startfile(video_path)
        elif system == 'Darwin':  # macOS
            subprocess.call(['open', video_path])
        else:  # Linux and other Unix-like systems
            subprocess.call(['xdg-open', video_path])
        
        logger.info(f"Opened video: {video_path}")
    except Exception as e:
        logger.error(f"Error opening video with system viewer: {e}")
        # Continue execution even if display fails

def get_video_output_folder(video_path: str, operation: str) -> str:
    """
    Create and return a folder for storing video processing outputs
    
    Args:
        video_path: Path to the video file
        operation: Name of operation being performed
        
    Returns:
        str: Path to the output folder
    """
    # Get directory of original video
    directory = os.path.dirname(video_path) or '.'
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create folder name with video name, operation and timestamp
    timestamp = get_timestamp()
    folder_name = f"{video_name}_{operation}_{timestamp}"
    folder_path = os.path.join(directory, folder_name)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def save_and_display(img: np.ndarray, original_path: str, operation: str) -> str:
    """
    Save image to file and display it using system's default image viewer
    
    Args:
        img: OpenCV image
        original_path: Path to original image
        operation: Name of operation performed
        
    Returns:
        str: Path to saved image
    """
    # Determine if this is a video frame by checking if the path contains specific markers
    is_video_frame = any(marker in os.path.basename(original_path) for marker in 
                         ["_frame_", "_track_", "_motion_"])
    
    # Get filename without extension
    base_name = os.path.basename(original_path)
    name_parts = os.path.splitext(base_name)
    
    # Create new filename with operation and timestamp
    timestamp = get_timestamp()
    new_filename = f"{name_parts[0]}_{operation}_{timestamp}{name_parts[1]}"
    
    # Get directory based on whether it's a video frame or regular image
    if is_video_frame:
        # Use the same directory as the original
        directory = os.path.dirname(original_path)
    else:
        # Get directory of original image
        directory = os.path.dirname(original_path) or '.'
    
    new_path = str(safe_path(os.path.join(directory, new_filename)))
    
    # Save image
    cv2.imwrite(new_path, img)
    
    # Display image only when explicitly enabled.
    open_image_with_system_viewer(new_path)
    
    return new_path
