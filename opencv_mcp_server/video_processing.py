from __future__ import annotations

import cv2
import numpy as np
import os
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, Tuple

from .mcp_instance import mcp
from .utils.contracts import success_response, error_response
from .utils.path_utils import safe_path
from .utils.config import (
    ENABLE_CAMERA,
    MAX_CAMERA_DURATION_SECONDS,
    MAX_IMAGE_DIMENSION,
    MAX_VIDEO_FPS,
    MAX_VIDEO_FRAMES,
    get_models_dir,
)
from .utils.cv_utils import (
    get_image_info,
    get_timestamp,
    validate_float_param,
    validate_int_param,
)

logger = logging.getLogger("merlin-cv-mcp.video_processing")

def get_video_info_sync(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            "width": width, "height": height, "fps": float(fps),
            "frame_count": frame_count, "duration_seconds": float(duration)
        }
    finally:
        cap.release()

@mcp.tool()
async def extract_video_frames_tool(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    max_frames: int = 10
) -> Dict[str, Any]:
    """
    Extracts frames from a video at specific intervals.
    """
    try:
        vp = str(safe_path(video_path))
        info = await asyncio.to_thread(get_video_info_sync, vp)
        
        step = validate_int_param("step", step, minimum=1)
        max_f = validate_int_param("max_frames", max_frames, minimum=1, maximum=MAX_VIDEO_FRAMES)
        
        def extract_sync(path, start, end, stp, limit, fps):
            cap = cv2.VideoCapture(path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                frames = []
                count = 0
                curr = start
                while count < limit and (end is None or curr <= end):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Save frame
                    f_name = f"{os.path.splitext(os.path.basename(path))[0]}_frame_{curr}.jpg"
                    f_path = os.path.join(os.path.dirname(path), f_name)
                    cv2.imwrite(f_path, frame)
                    
                    frames.append({
                        "index": curr,
                        "timestamp": curr / fps if fps > 0 else 0,
                        "path": f_path
                    })
                    count += 1
                    curr += stp
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
                return frames
            finally:
                cap.release()

        frames = await asyncio.to_thread(extract_sync, vp, start_frame, end_frame, step, max_f, info["fps"])
        return success_response({"frames": frames, "video_info": info}, tool_name="extract_video_frames_tool")
    except Exception as e:
        return error_response(str(e), tool_name="extract_video_frames_tool")

@mcp.tool()
async def detect_motion_tool(
    frame1_path: str,
    frame2_path: str,
    threshold: int = 25,
    min_area: int = 500
) -> Dict[str, Any]:
    """
    Detects significant motion between two extracted frames.
    """
    try:
        p1 = str(safe_path(frame1_path))
        p2 = str(safe_path(frame2_path))
        
        def motion_sync(path1, path2, th, ma):
            f1 = cv2.imread(path1)
            f2 = cv2.imread(path2)
            if f1 is None or f2 is None: raise ValueError("Failed to read frames")
            
            g1 = cv2.GaussianBlur(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            g2 = cv2.GaussianBlur(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            
            diff = cv2.absdiff(g1, g2)
            _, thresh = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motions = []
            f2_res = f2.copy()
            for c in contours:
                if cv2.contourArea(c) < ma: continue
                (x, y, w, h) = cv2.boundingRect(c)
                motions.append({"bbox": [int(x), int(y), int(w), int(h)], "area": float(cv2.contourArea(c))})
                cv2.rectangle(f2_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            res_path = os.path.join(os.path.dirname(path2), f"motion_{get_timestamp()}.jpg")
            cv2.imwrite(res_path, f2_res)
            return motions, res_path

        motions, path = await asyncio.to_thread(motion_sync, p1, p2, threshold, min_area)
        return success_response({"motion_detected": len(motions) > 0, "motions": motions, "path": path}, tool_name="detect_motion_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_motion_tool")

@mcp.tool()
async def combine_frames_to_video_tool(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Combines a sequence of static images into a video file.
    """
    try:
        fps = validate_float_param("fps", fps, minimum=0.1, maximum=MAX_VIDEO_FPS)
        p_out = str(safe_path(output_path))
        p_frames = [str(safe_path(f)) for f in frame_paths]
        
        if len(p_frames) > MAX_VIDEO_FRAMES:
            return error_response(f"Too many frames. Limit is {MAX_VIDEO_FRAMES}")

        def combine_sync(frames, out_p, f_rate):
            first = cv2.imread(frames[0])
            if first is None: raise ValueError(f"Failed to read first frame: {frames[0]}")
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_p, fourcc, f_rate, (w, h))
            try:
                if not writer.isOpened(): raise ValueError(f"Failed to open VideoWriter for {out_p}")
                for f in frames:
                    img = cv2.imread(f)
                    if img is not None:
                        if img.shape[0] != h or img.shape[1] != w:
                            img = cv2.resize(img, (w, h))
                        writer.write(img)
                return out_p
            finally:
                writer.release()

        res_path = await asyncio.to_thread(combine_sync, p_frames, p_out, fps)
        return success_response({"path": res_path}, tool_name="combine_frames_to_video_tool")
    except Exception as e:
        return error_response(str(e), tool_name="combine_frames_to_video_tool")
