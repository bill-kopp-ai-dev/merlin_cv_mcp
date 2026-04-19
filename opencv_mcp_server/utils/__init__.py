from .cv_utils import (
    get_image_info,
    get_timestamp,
    save_and_display,
    validate_float_param,
    validate_int_param,
)
from .path_utils import safe_path
from .config import get_models_dir

# For backward compatibility if needed
def resolve_model_asset_path(requested_path, default_filename):
    from pathlib import Path
    model_dir = get_models_dir()
    raw_path = (requested_path or "").strip()
    if not raw_path:
        target_path = (model_dir / default_filename).resolve()
    else:
        target_path = Path(raw_path).expanduser().resolve()
    if not target_path.is_relative_to(model_dir):
        raise PermissionError(f"Path {target_path} outside {model_dir}")
    return str(target_path)

def sanitize_class_label(raw_label: str, fallback: str = "unknown") -> str:
    import re
    original = "" if raw_label is None else str(raw_label)
    text = original.replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"[^A-Za-z0-9 _./:+()-]+", " ", text)
    return " ".join(text.split()) or fallback
