from __future__ import annotations

import cv2
import numpy as np
import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Union

from .mcp_instance import mcp
from .utils.contracts import success_response, error_response
from .utils.path_utils import safe_path
from .utils.cv_utils import (
    get_image_info,
    save_and_display,
)

logger = logging.getLogger("merlin-cv-mcp.image_processing")

@mcp.tool()
async def apply_filter_tool(
    image_path: str, 
    filter_type: str, 
    kernel_size: Union[int, List[int]], 
    sigma: Optional[float] = None,
    sigma_color: Optional[float] = None,
    sigma_space: Optional[float] = None
) -> Dict[str, Any]:
    """
    Applies blur filters (blur, gaussian, median, bilateral) to an image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_filterSync(img, f_type, k_size, s, sc, ss):
            # Ensure kernel size is odd and formatted correctly
            if isinstance(k_size, int):
                if k_size % 2 == 0: k_size += 1
                k_size = (k_size, k_size)
            elif isinstance(k_size, (list, tuple)):
                k_size = (k_size[0] if k_size[0] % 2 != 0 else k_size[0]+1, 
                          k_size[1] if k_size[1] % 2 != 0 else k_size[1]+1)

            if f_type.lower() == 'blur':
                result = cv2.blur(img, k_size)
                info = {"type": "blur", "kernel_size": k_size}
            elif f_type.lower() == 'gaussian':
                result = cv2.GaussianBlur(img, k_size, s or 0)
                info = {"type": "gaussian", "kernel_size": k_size, "sigma": s or 0}
            elif f_type.lower() == 'median':
                ks = max(k_size)
                if ks % 2 == 0: ks += 1
                result = cv2.medianBlur(img, ks)
                info = {"type": "median", "kernel_size": ks}
            elif f_type.lower() == 'bilateral':
                d = max(k_size)
                result = cv2.bilateralFilter(img, d, sc or 75, ss or 75)
                info = {"type": "bilateral", "diameter": d, "sigma_color": sc or 75, "sigma_space": ss or 75}
            else:
                raise ValueError(f"Unsupported filter type: {f_type}")
            return result, info

        result, filter_info = await asyncio.to_thread(process_filterSync, img, filter_type, kernel_size, sigma, sigma_color, sigma_space)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"filter_{filter_type}")
        
        data = {
            "filter": filter_info,
            "info": get_image_info(result),
            "path": new_path,
            "output_path": new_path
        }
        return success_response(data, tool_name="apply_filter_tool")
    except Exception as e:
        return error_response(str(e), tool_name="apply_filter_tool")

@mcp.tool()
async def detect_edges_tool(
    image_path: str, 
    method: str = "canny", 
    threshold1: float = 100.0, 
    threshold2: float = 200.0,
    aperture_size: int = 3,
    l2gradient: bool = False,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0
) -> Dict[str, Any]:
    """
    Detects boundaries and edges in an image (canny, sobel, laplacian, scharr).
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_edgesSync(img, m, t1, t2, ap, l2, ks, sc, dl):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            if m.lower() == 'canny':
                edges = cv2.Canny(gray, t1, t2, apertureSize=ap, L2gradient=l2)
            elif m.lower() == 'sobel':
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ks, scale=sc, delta=dl)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ks, scale=sc, delta=dl)
                mag = cv2.magnitude(sx, sy)
                edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            elif m.lower() == 'laplacian':
                lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ks, scale=sc, delta=dl)
                edges = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:
                raise ValueError(f"Unsupported edge method: {m}")
            return edges

        edges = await asyncio.to_thread(process_edgesSync, img, method, threshold1, threshold2, aperture_size, l2gradient, ksize, scale, delta)
        new_path = await asyncio.to_thread(save_and_display, edges, p, f"edges_{method}")
        
        return success_response({"info": get_image_info(edges), "path": new_path, "output_path": new_path}, tool_name="detect_edges_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_edges_tool")

@mcp.tool()
async def apply_threshold_tool(
    image_path: str, 
    threshold_type: str = "binary", 
    threshold_value: float = 127.0, 
    max_value: float = 255.0,
    adaptive_method: str = "gaussian",
    block_size: int = 11,
    c: float = 2.0
) -> Dict[str, Any]:
    """
    Applies thresholding, converting the image to black or white pixels.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
            
        def process_threshSync(img, tt, tv, mv, am, bs, c_val):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            if tt.lower() == 'adaptive':
                bs = bs if bs % 2 != 0 else bs + 1
                method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if am.lower() == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
                result = cv2.adaptiveThreshold(gray, mv, method, cv2.THRESH_BINARY, bs, c_val)
            else:
                tt_map = {"binary": cv2.THRESH_BINARY, "binary_inv": cv2.THRESH_BINARY_INV, "trunc": cv2.THRESH_TRUNC}
                code = tt_map.get(tt.lower(), cv2.THRESH_BINARY)
                _, result = cv2.threshold(gray, tv, mv, code)
            return result

        result = await asyncio.to_thread(process_threshSync, img, threshold_type, threshold_value, max_value, adaptive_method, block_size, c)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"threshold_{threshold_type}")
        return success_response({"path": new_path, "output_path": new_path}, tool_name="apply_threshold_tool")
    except Exception as e:
        return error_response(str(e), tool_name="apply_threshold_tool")

@mcp.tool()
async def detect_contours_tool(
    image_path: str, 
    mode: str = "external", 
    method: str = "simple",
    draw: bool = True,
    thickness: int = 1,
    color: List[int] = [0, 255, 0],
    threshold_value: float = 127.0
) -> Dict[str, Any]:
    """
    Detects and draws contour lines in an image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_contoursSync(img, m, meth, d, thick, clr, tv):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
            
            modes = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE}
            meths = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE}
            
            contours, _ = cv2.findContours(binary, modes.get(m.lower(), cv2.RETR_EXTERNAL), meths.get(meth.lower(), cv2.CHAIN_APPROX_SIMPLE))
            
            img_c = img.copy()
            if d:
                cv2.drawContours(img_c, contours, -1, tuple(clr), thick)
            
            return img_c, len(contours)

        result, count = await asyncio.to_thread(process_contoursSync, img, mode, method, draw, thickness, color, threshold_value)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"contours_{count}")
        return success_response({"contour_count": count, "path": new_path, "output_path": new_path}, tool_name="detect_contours_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_contours_tool")

@mcp.tool()
async def find_shapes_tool(
    image_path: str, 
    shape_type: str, 
    param1: float = 100.0, 
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0,
    min_dist: int = 50,
    threshold: float = 150.0,
    draw: bool = True,
    color: List[int] = [0, 0, 255]
) -> Dict[str, Any]:
    """
    Finds and draws basic geometric shapes (circles, lines).
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_shapesSync(img, st, p1, p2, mir, mar, md, th, d, clr):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            img_c = img.copy()
            shapes = []
            
            if st.lower() == 'circles':
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=md, param1=p1, param2=p2, minRadius=mir, maxRadius=mar)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i, circle in enumerate(circles[0, :]):
                        center = (int(circle[0]), int(circle[1]))
                        radius = int(circle[2])
                        shapes.append({"index": i, "center": center, "radius": radius})
                        if d: cv2.circle(img_c, center, radius, tuple(clr), 2)
            return img_c, shapes

        result, shapes = await asyncio.to_thread(process_shapesSync, img, shape_type, param1, param2, min_radius, max_radius, min_dist, threshold, draw, color)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"shapes_{shape_type}")
        return success_response({"shape_count": len(shapes), "shapes": shapes, "path": new_path, "output_path": new_path}, tool_name="find_shapes_tool")
    except Exception as e:
        return error_response(str(e), tool_name="find_shapes_tool")

@mcp.tool()
async def match_template_tool(
    image_path: str, 
    template_path: str, 
    method: str = "ccoeff_normed",
    threshold: float = 0.8,
    draw: bool = True
) -> Dict[str, Any]:
    """
    Locates exact template within a larger image.
    """
    try:
        p_img = str(safe_path(image_path))
        p_tpl = str(safe_path(template_path))
        img = await asyncio.to_thread(cv2.imread, p_img)
        tpl = await asyncio.to_thread(cv2.imread, p_tpl)
        if img is None or tpl is None:
            return error_response("Failed to read image or template", code="file_not_found")
        
        def process_matchSync(img, tpl, m, th, d):
            h, w = tpl.shape[:2]
            meth_map = {"sqdiff": cv2.TM_SQDIFF, "ccorr": cv2.TM_CCORR, "ccoeff_normed": cv2.TM_CCOEFF_NORMED}
            meth = meth_map.get(m.lower(), cv2.TM_CCOEFF_NORMED)
            
            res = cv2.matchTemplate(img, tpl, meth)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            best_loc = max_loc if meth not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else min_loc
            best_val = max_val if meth not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else 1.0 - min_val
            
            matches = []
            if best_val >= th:
                matches.append({"top_left": list(best_loc), "confidence": float(best_val)})
                if d:
                    cv2.rectangle(img, best_loc, (best_loc[0] + w, best_loc[1] + h), (0, 255, 0), 2)
            return img, matches

        result, matches = await asyncio.to_thread(process_matchSync, img.copy(), tpl, method, threshold, draw)
        new_path = await asyncio.to_thread(save_and_display, result, p_img, "template_match")
        return success_response({"match_count": len(matches), "matches": matches, "path": new_path, "output_path": new_path}, tool_name="match_template_tool")
    except Exception as e:
        return error_response(str(e), tool_name="match_template_tool")