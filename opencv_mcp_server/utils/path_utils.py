from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import ALLOWED_ROOTS, STRICT_PATH_VALIDATION

logger = logging.getLogger("merlin-cv-mcp.path_utils")

def safe_path(requested_path: str, working_dir: Optional[str] = None) -> Path:
    """
    Enforces that the requested path is within one of the ALLOWED_ROOTS.
    Supports '~' for home directory and resolves relative paths.
    """
    # 1. Expand user and resolve
    raw_path = Path(requested_path).expanduser()
    
    # 2. Handle working_dir if provided for relative paths
    if not raw_path.is_absolute():
        if working_dir:
            base = Path(working_dir).expanduser().resolve()
            target_path = (base / raw_path).resolve()
        else:
            # Default to CWD or a safe default if no working_dir
            target_path = raw_path.resolve()
    else:
        target_path = raw_path.resolve()

    if not STRICT_PATH_VALIDATION:
        return target_path

    # 3. Check against ALLOWED_ROOTS
    is_safe = False
    for root in ALLOWED_ROOTS:
        try:
            if target_path.is_relative_to(root):
                is_safe = True
                break
        except ValueError:
            continue

    if not is_safe:
        msg = f"Access denied: Path {target_path} is outside allowed roots."
        logger.warning(msg)
        raise PermissionError(msg)
        
    return target_path

def is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except (ValueError, AttributeError):
        return False
