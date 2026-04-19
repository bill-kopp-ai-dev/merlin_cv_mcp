from __future__ import annotations

import logging
from typing import Any, Dict

from .mcp_instance import mcp
from .utils.contracts import success_response

logger = logging.getLogger("merlin-cv-mcp.profile")

@mcp.tool()
async def get_merlin_cv_profile() -> Dict[str, Any]:
    """
    Returns the operational profile and capabilities of the Merlin CV server.
    
    USO ESTRATÉGICO:
    - O agente chama esta ferramenta para entender quais algoritmos de visão (YOLO, Haar, etc) 
      estão ativos e quais os limites de hardware (resolução, FPS) do ambiente atual.
    """
    profile = {
        "server": "merlin-cv-mcp",
        "version": "1.0.0",
        "description": "High-performance OpenCV manipulation server",
        "capabilities": [
            "image_manipulation",
            "feature_detection",
            "face_detection",
            "object_detection_yolo",
            "video_frame_extraction",
            "motion_analysis"
        ],
        "frameworks": {
            "opencv": "4.x",
            "mcp": "FastMCP"
        },
        "status": "operational"
    }
    return success_response(profile, tool_name="get_merlin_cv_profile")
