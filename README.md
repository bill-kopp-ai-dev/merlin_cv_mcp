# Merlin CV MCP Server 🧙‍♂️

A high-performance Model Context Protocol (MCP) server for local image and video manipulation using **OpenCV**, integrated with the **Percival OS** ecosystem.

## Features

- **Async Architecture**: Non-blocking I/O using `asyncio.to_thread` for heavy CPU-bound OpenCV operations.
- **Image Basics**: Crop, Resize, Color Conversion, and detailed Statistical Analysis.
- **Advanced Image Processing**: Edge detection (Canny/Sobel), Thresholding, Contour detection, and Template Matching.
- **Computer Vision**: Face detection (Haar/DNN) and Object Detection (YOLOv3).
- **Video Handling**: Frame extraction, Motion detection, and Video creation from frames.
- **Agnostic Sandbox**: Comprehensive path validation supporting cross-platform `~/` (Home) and workspace paths.
- **Security Telemetry**: Real-time metrics for access control and sanitization events.

## Installation

```bash
uv pip install -e .
```

## Configuration

The server supports several environment variables to tune performance and security:

- `NANOBOT_WORKSPACE`: Root directory for file operations.
- `MERLIN_CV_MAX_IMAGE_DIMENSION`: Max pixel width/height (Default: 4096).
- `MERLIN_CV_MAX_VIDEO_FRAMES`: Limit for video extraction/creation (Default: 1200).
- `OPENCV_DNN_MODELS_DIR`: Path to YOLO and Face detection models.

## Tools Summary

### Image
- `save_image_tool`: Duplicate or rename images.
- `resize_image_tool`: Scale images to avoid "Payload Too Large" errors.
- `crop_image_tool`: Focus on specific regions for better LLM analysis.
- `apply_filter_tool`: Smooth or sharpen images.
- `detect_objects_tool`: Fast local object detection using YOLO.

### Video
- `extract_video_frames_tool`: Convert video segments to images for Vision LLMs.
- `detect_motion_tool`: Efficiently find where things move before expensive analysis.
- `combine_frames_to_video_tool`: Create annotated videos from processed frames.

### Ecosystem Integration
- `get_merlin_cv_profile`: Return machine-readable server profile and hardware capabilities for Nanobot orchestration.
- `get_security_metrics`: Real-time audit counters for security events.

## Model Assets

To use object detection and face DNN tools, place the following in your `models/` directory:
- `yolov3.weights` & `yolov3.cfg` (from PJReddie)
- `coco.names`
- `deploy.prototxt` & `res10_300x300_ssd_iter_140000.caffemodel` (Face ResNet)

## License

MIT - Percival Ecosystem
