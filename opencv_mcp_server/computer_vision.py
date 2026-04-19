from __future__ import annotations

import cv2
import numpy as np
import os
import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Union

from .mcp_instance import mcp
from .utils.contracts import success_response, error_response
from .utils.path_utils import safe_path
from .utils.config import get_max_image_dimension, get_models_dir
from .utils.cv_utils import (
    get_image_info,
    save_and_display,
    validate_float_param,
    validate_int_param,
)

logger = logging.getLogger("merlin-cv-mcp.computer_vision")

_YOLO_MODEL_CACHE: Dict[Tuple[str, str], Tuple[cv2.dnn_Net, List[str]]] = {}
_CLASS_NAME_CACHE: Dict[str, List[str]] = {}

def _resolve_yolo_output_layers(net: cv2.dnn_Net) -> List[str]:
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def _load_class_names(classes_path: str) -> List[str]:
    cached = _CLASS_NAME_CACHE.get(classes_path)
    if cached is not None:
        return cached
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    _CLASS_NAME_CACHE[classes_path] = classes
    return classes

def _load_yolo_net(model_path: str, config_path: str) -> Tuple[cv2.dnn_Net, List[str]]:
    cache_key = (model_path, config_path)
    cached = _YOLO_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    net = cv2.dnn.readNetFromDarknet(config_path, model_path)
    output_layers = _resolve_yolo_output_layers(net)
    _YOLO_MODEL_CACHE[cache_key] = (net, output_layers)
    return net, output_layers

@mcp.tool()
async def detect_features_tool(
    image_path: str, 
    method: str = "sift", 
    max_features: int = 500,
    draw: bool = True
) -> Dict[str, Any]:
    """
    Finds keypoints and vital characteristics in an image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_featuresSync(img, m, mf, d):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            m_low = m.lower()
            if m_low == 'sift':
                detector = cv2.SIFT_create(nfeatures=mf)
            elif m_low == 'orb':
                detector = cv2.ORB_create(nfeatures=mf)
            else:
                raise ValueError(f"Unsupported feature method: {m}")
            
            kp, des = detector.detectAndCompute(gray, None)
            res_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) if d else img
            return res_img, len(kp)

        result, count = await asyncio.to_thread(process_featuresSync, img, method, max_features, draw)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"features_{method}")
        return success_response({"keypoint_count": count, "path": new_path, "output_path": new_path}, tool_name="detect_features_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_features_tool")

@mcp.tool()
async def detect_faces_tool(
    image_path: str, 
    method: str = "haar", 
    scale_factor: float = 1.3,
    min_neighbors: int = 5
) -> Dict[str, Any]:
    """
    Detects human faces in an image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        def process_facesSync(img, m, sf, mn):
            faces = []
            img_c = img.copy()
            if m.lower() == 'haar':
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if face_cascade.empty():
                    raise RuntimeError("Failed to load Haar Cascade. Check OpenCV installation.")
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                rects = face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30))
                for i, (x, y, w, h) in enumerate(rects):
                    faces.append({"index": i, "x": int(x), "y": int(y), "width": int(w), "height": int(h)})
                    cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                raise ValueError(f"Unsupported face detection method: {m}")
            return img_c, faces

        result, faces = await asyncio.to_thread(process_facesSync, img, method, scale_factor, min_neighbors)
        new_path = await asyncio.to_thread(save_and_display, result, p, f"faces_{len(faces)}")
        return success_response({"face_count": len(faces), "faces": faces, "path": new_path, "output_path": new_path}, tool_name="detect_faces_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_faces_tool")

@mcp.tool()
async def detect_objects_tool(
    image_path: str, 
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Uses YOLOv3 to identify and mark basic object classes in the image.
    """
    try:
        p = str(safe_path(image_path))
        img = await asyncio.to_thread(cv2.imread, p)
        if img is None:
            return error_response(f"Failed to read image: {image_path}", code="file_not_found")
        
        orig_h, orig_w = img.shape[:2]
        model_dir = str(get_models_dir())
        m_path = os.path.join(model_dir, "yolov3.weights")
        c_path = os.path.join(model_dir, "yolov3.cfg")
        cl_path = os.path.join(model_dir, "coco.names")

        if not (os.path.exists(m_path) and os.path.exists(c_path)):
            return error_response("YOLO model files not found. Please download yolov3.weights and yolov3.cfg to models/", code="model_not_found")

        def process_yoloSync(img, m_p, c_p, cl_p, conf_t, nms_t):
            classes = _load_class_names(cl_p)
            net, output_layers = _load_yolo_net(m_p, c_p)
            
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(output_layers)
            
            boxes, confs, class_ids = [], [], []
            for out in layer_outputs:
                for det in out:
                    scores = det[5:]
                    c_id = np.argmax(scores)
                    conf = scores[c_id]
                    if conf > conf_t:
                        cx, cy = int(det[0] * orig_w), int(det[1] * orig_h)
                        bw, bh = int(det[2] * orig_w), int(det[3] * orig_h)
                        bx, by = int(cx - bw/2), int(cy - bh/2)
                        boxes.append([bx, by, bw, bh])
                        confs.append(float(conf))
                        class_ids.append(c_id)
            
            indices = cv2.dnn.NMSBoxes(boxes, confs, conf_t, nms_t)
            objects = []
            img_c = img.copy()
            if len(indices) > 0:
                if isinstance(indices, np.ndarray): indices = indices.flatten().tolist()
                for i in indices:
                    idx = int(i)
                    x, y, w, h = boxes[idx]
                    cn = classes[class_ids[idx]] if class_ids[idx] < len(classes) else f"id_{class_ids[idx]}"
                    objects.append({"class": cn, "confidence": confs[idx], "bbox": [x, y, w, h]})
                    cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_c, f"{cn} {confs[idx]:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return img_c, objects

        result, objects = await asyncio.to_thread(process_yoloSync, img, m_path, c_path, cl_path, confidence_threshold, nms_threshold)
        new_path = await asyncio.to_thread(save_and_display, result, p, "yolo_detected")
        return success_response({"object_count": len(objects), "objects": objects, "path": new_path, "output_path": new_path}, tool_name="detect_objects_tool")
    except Exception as e:
        return error_response(str(e), tool_name="detect_objects_tool")
