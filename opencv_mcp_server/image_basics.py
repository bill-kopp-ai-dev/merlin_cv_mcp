from __future__ import annotations

import cv2
import numpy as np
import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, Union

from .mcp_instance import mcp
from .utils.config import MAX_IMAGE_DIMENSION
from .utils.contracts import success_response, error_response
from .utils.path_utils import safe_path
from .utils.cv_utils import (
    get_image_info,
    save_and_display,
    validate_int_param,
)

logger = logging.getLogger("merlin-cv-mcp.image_basics")

@mcp.tool()
async def save_image_tool(path_in: str, path_out: str) -> Dict[str, Any]:
    """
    Saves a copy of an image or creates a new file from another image.
    """
    try:
        p_in = str(safe_path(path_in))
        p_out = str(safe_path(path_out))
        
        img = await asyncio.to_thread(cv2.imread, p_in)
        if img is None:
            return error_response(f"Failed to read image from path: {path_in}", code="file_not_found")
        
        directory = os.path.dirname(p_out)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        new_path = await asyncio.to_thread(save_and_display, img, p_out, "save")
        
        data = {
            "success": True,
            "path": new_path,
            "size_bytes": os.path.getsize(new_path),
            "output_path": new_path
        }
        return success_response(data, tool_name="save_image_tool")
    except Exception as e:
        return error_response(str(e), tool_name="save_image_tool")

@mcp.tool()
async def convert_color_space_tool(image_path: str, source_space: str, target_space: str) -> Dict[str, Any]:
    """
    Converts an image between different color spaces (e.g., BGR to GRAY).
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        # Internal map for BGR -> Target
        # Note: OpenCV uses 'HLS' internally for what many call 'HSL'.
        color_space_map = {
            "RGB": cv2.COLOR_BGR2RGB,
            "GRAY": cv2.COLOR_BGR2GRAY,
            "HSV": cv2.COLOR_BGR2HSV,
            "LAB": cv2.COLOR_BGR2LAB,
            "YCrCb": cv2.COLOR_BGR2YCrCb,
            "XYZ": cv2.COLOR_BGR2XYZ,
            "HLS": cv2.COLOR_BGR2HLS,
            "HSL": cv2.COLOR_BGR2HLS, # Alias for HLS
            "YUV": cv2.COLOR_BGR2YUV,
        }
        
        # Reverse map for Target -> BGR (for display/saving)
        reverse_map = {
            "RGB": cv2.COLOR_RGB2BGR,
            "GRAY": cv2.COLOR_GRAY2BGR,
            "HSV": cv2.COLOR_HSV2BGR,
            "LAB": cv2.COLOR_LAB2BGR,
            "YCrCb": cv2.COLOR_YCrCb2BGR,
            "XYZ": cv2.COLOR_XYZ2BGR,
            "HLS": cv2.COLOR_HLS2BGR,
            "HSL": cv2.COLOR_HLS2BGR, # Alias
            "YUV": cv2.COLOR_YUV2BGR,
        }
        
        s_space = source_space.upper()
        t_space = target_space.upper()
        
        if s_space == "BGR" and t_space == "BGR":
            new_path = await asyncio.to_thread(save_and_display, img, p, "convert_BGR_to_BGR")
            return success_response({"path": new_path}, tool_name="convert_color_space_tool")

        if s_space == "BGR" and t_space in color_space_map:
            conversion_code = color_space_map[t_space]
            converted = await asyncio.to_thread(cv2.cvtColor, img, conversion_code)
            
            # For display purposes (saving as JPG/PNG), we ensure it's BGR
            display_img = converted
            if t_space != "BGR":
                back_code = reverse_map.get(t_space)
                if back_code is not None:
                    display_img = await asyncio.to_thread(cv2.cvtColor, converted, back_code)
            
            new_path = await asyncio.to_thread(save_and_display, display_img, p, f"convert_{s_space}_to_{t_space}")
            
            data = {
                "source_space": s_space,
                "target_space": t_space,
                "info": get_image_info(converted),
                "path": new_path,
                "output_path": new_path
            }
            return success_response(data, tool_name="convert_color_space_tool")
        else:
            return error_response(f"Unsupported conversion: {source_space} to {target_space}", code="unsupported_conversion")
            
    except Exception as e:
        return error_response(str(e), tool_name="convert_color_space_tool")

@mcp.tool()
async def resize_image_tool(image_path: str, width: int, height: int, interpolation: str = "INTER_LINEAR") -> Dict[str, Any]:
    """
    Resizes an image to exact dimensions in pixels.
    """
    try:
        p = str(safe_path(image_path))
        width = validate_int_param("width", width, minimum=1, maximum=MAX_IMAGE_DIMENSION)
        height = validate_int_param("height", height, minimum=1, maximum=MAX_IMAGE_DIMENSION)
        
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        interp_methods = {
            "INTER_NEAREST": cv2.INTER_NEAREST,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4
        }
        interp = interp_methods.get(interpolation.upper(), cv2.INTER_LINEAR)
        
        resized = await asyncio.to_thread(cv2.resize, img, (width, height), interpolation=interp)
        new_path = await asyncio.to_thread(save_and_display, resized, p, f"resize_{width}x{height}")
        
        data = {
            "width": width,
            "height": height,
            "interpolation": interpolation,
            "info": get_image_info(resized),
            "original_info": get_image_info(img),
            "path": new_path,
            "output_path": new_path
        }
        return success_response(data, tool_name="resize_image_tool")
    except Exception as e:
        return error_response(str(e), tool_name="resize_image_tool")

@mcp.tool()
async def crop_image_tool(image_path: str, x: int, y: int, width: int, height: int) -> Dict[str, Any]:
    """
    Crops a specific region of an image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        img_h, img_w = img.shape[:2]
        x = validate_int_param("x", x, minimum=0, maximum=img_w-1)
        y = validate_int_param("y", y, minimum=0, maximum=img_h-1)
        width = validate_int_param("width", width, minimum=1, maximum=img_w-x)
        height = validate_int_param("height", height, minimum=1, maximum=img_h-y)
        
        cropped = img[y:y+height, x:x+width]
        new_path = await asyncio.to_thread(save_and_display, cropped, p, f"crop_{x}_{y}")
        
        data = {
            "x": x, "y": y, "width": width, "height": height,
            "info": get_image_info(cropped),
            "path": new_path,
            "output_path": new_path
        }
        return success_response(data, tool_name="crop_image_tool")
    except Exception as e:
        return error_response(str(e), tool_name="crop_image_tool")

@mcp.tool()
async def get_image_stats_tool(image_path: str, channels: bool = True) -> Dict[str, Any]:
    """
    Calculates statistical information about an image (histogram, mean, stddev).
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def calculate_statsSync(img, channels):
            mean, stddev = cv2.meanStdDev(img)
            stats = {
                "info": get_image_info(img),
                "min": float(np.min(img)),
                "max": float(np.max(img)),
                "mean": float(np.mean(img)),
                "stddev": float(np.std(img))
            }
            if channels and len(img.shape) > 2:
                c_stats = []
                for i in range(img.shape[2]):
                    c = img[:,:,i]
                    c_stats.append({
                        "channel": i,
                        "min": float(np.min(c)), "max": float(np.max(c)),
                        "mean": float(np.mean(c)), "stddev": float(np.std(c))
                    })
                stats["channels"] = c_stats
            return stats

        stats = await asyncio.to_thread(calculate_statsSync, img, channels)
        return success_response(stats, tool_name="get_image_stats_tool")
    except Exception as e:
        return error_response(str(e), tool_name="get_image_stats_tool")
