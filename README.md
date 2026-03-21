# 🧙 Merlin CV MCP Server

**Merlin** is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server focused on fast, offline image and video manipulation using OpenCV. It acts as the local editing layer of the [nanobot](https://github.com/HKUDS/nanobot) autonomous agent ecosystem within [percival.OS](https://github.com/bill-kopp-ai-dev/percival.OS_Dev), enabling the LLM to preprocess images before sending them to vision APIs or post-process images after generation.

---

## 🙏 Credits & Original Repository

This project is a refactored fork of **[opencv-mcp-server](https://github.com/GongRzhe/opencv-mcp-server)**, originally created by **GongRzhe**.

The core pixel-matrix processing tools (resize, crop, color conversion, filters, edge detection) are derived from the original project. Our refactoring focused on three pillars essential for autonomous agents: **Security (Sandboxing)**, **Pipeline Intelligence (Orchestration)**, and **Infrastructure Standardization**.

---

## 🛠️ What Changed? (Refactoring Details)

The following architectural improvements were implemented to transform `opencv-mcp-server` into **Merlin**:

### 1. Shared Workspace (Sandboxing)

The original server allowed OpenCV to read and write to any directory on the operating system — a serious security risk (e.g. Directory Traversal attacks) when controlled by an LLM.

- **Change:** Implemented the `safe_path()` utility and the `NANOBOT_WORKSPACE` environment variable. All file operations are validated against this sandbox boundary before execution.
- **Benefit:** Merlin now operates in a fully isolated environment. It can only read and save images within the agent's shared workspace directory, ensuring complete security for the host machine.

### 2. Pipeline Intelligence (Semantic Docstrings)

An autonomous agent needs to know *when* to use a tool, not just *what* it does.

- **Change:** Docstrings for all tools were rewritten to include "Pipeline Context" describing the orchestration role of each tool.
- **Benefit:** The LLM now understands that it should `resize_image` *before* sending large files to vision APIs (preventing payload errors), or that it should `crop_image` + `convert_color_space` to isolate a license plate and improve contrast *before* triggering an OCR tool from another MCP server.

### 3. Headless Infrastructure & Standardization

Traditional Python OpenCV setups often hit GUI dependency issues in servers and Docker containers.

- **Change:** Removed legacy installation scripts and migrated to `uv` + `pyproject.toml`. Replaced the standard `opencv-python` with `opencv-python-headless`.
- **Benefit:** Merlin runs perfectly in cloud environments, Docker containers, and headless Linux servers without GUI libraries (no `libGL` errors).

---

## 🔌 MCP Tools

### 🖼️ Image Basics (`image_basics`)

| Tool | Description |
|---|---|
| `save_image_tool` | Copy/save an image from one path to another within the workspace |
| `resize_image_tool` | Resize an image to specified dimensions with configurable interpolation |
| `crop_image_tool` | Crop a rectangular region from an image (x, y, width, height) |
| `convert_color_space_tool` | Convert between color spaces (RGB, BGR, HSV, Grayscale, LAB, etc.) |
| `get_image_stats_tool` | Get image statistics: dimensions, pixel value range, per-channel stats |

### ⚙️ Image Processing (`image_processing`)

| Tool | Description |
|---|---|
| `apply_filter_tool` | Apply blurring and sharpening filters (Gaussian, Median, Bilateral, etc.) |
| `detect_edges_tool` | Detect edges using Canny, Sobel, or Laplacian algorithms |
| `apply_threshold_tool` | Apply binary or adaptive thresholding for segmentation |
| `detect_contours_tool` | Find and return the contours of objects in an image |
| `find_shapes_tool` | Detect geometric shapes (circles, rectangles, triangles) |
| `match_template_tool` | Find a sub-image (template) within a larger image |

### 👁️ Computer Vision (`computer_vision`)

| Tool | Description |
|---|---|
| `detect_features_tool` | Detect and describe keypoints using ORB, SIFT, or AKAZE |
| `match_features_tool` | Match keypoints between two images for alignment/recognition |
| `detect_faces_tool` | Detect human faces using Haar Cascade classifiers |
| `detect_objects_tool` | Detect objects using a YOLO model from a configurable models directory |

### 🎬 Video Processing (`video_processing`)

| Tool | Description |
|---|---|
| `extract_video_frames_tool` | Extract frames from a video file at a given interval |
| `detect_motion_tool` | Detect motion between consecutive video frames |
| `track_object_tool` | Track a selected object across video frames |
| `combine_frames_to_video_tool` | Combine a sequence of image frames into a video file |
| `create_mp4_from_video_tool` | Re-encode a video to MP4 format |
| `detect_video_objects_tool` | Run YOLO object detection across video frames |

---

## 🚀 Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

---

## 📦 Installation

```bash
git clone https://github.com/bill-kopp-ai-dev/merlin_cv_mcp.git
cd merlin_cv_mcp
uv sync
```

---

## ▶️ Running

```bash
uv run opencv_mcp_server/main.py
```

Or via the installed script entry point:

```bash
merlin-cv
```

> **Important:** Make sure to set the `NANOBOT_WORKSPACE` environment variable pointing to your shared image directory.

---

## ⚙️ Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `NANOBOT_WORKSPACE` | ✅ | — | Absolute path to the shared workspace directory. All file operations are sandboxed to this path. |
| `OPENCV_DNN_MODELS_DIR` | ❌ | — | Path to directory containing YOLO model files (`.cfg`, `.weights`, `.names`) for `detect_objects_tool` and `detect_video_objects_tool` |

---

## 🤖 Integrating with nanobot / Claude Desktop

Add the following entry to your agent's `config.json`:

```json
"merlin-cv": {
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "opencv_mcp_server.main"],
  "env": {
    "PYTHONPATH": "/path/to/merlin_cv_mcp",
    "NANOBOT_WORKSPACE": "/path/to/shared/image/workspace",
    "OPENCV_DNN_MODELS_DIR": "/path/to/merlin_cv_mcp/models"
  }
}
```

### Pipeline Example

```
User: Read the license plate number in this photo.

Agent: [calls resize_image_tool to reduce to 800px wide — avoids large API payloads]
       [calls crop_image_tool to isolate the plate region]
       [calls convert_color_space_tool to convert to grayscale — improves OCR]
       [calls percival-vision → read_text with the preprocessed plate image]
       → "ABC-1234"

User: Generate a product image and apply a soft blur effect.

Agent: [calls jarvina → generate_image to create the product photo]
       [calls apply_filter_tool with Gaussian blur on the saved file]
       → Returns the processed image path
```

---

## 📁 Project Structure

```
merlin_cv_mcp/
├── pyproject.toml                        # Dependencies & script entry point
└── opencv_mcp_server/
    ├── main.py                           # Entry point — registers all tool modules
    ├── utils.py                          # safe_path() sandbox, image/video helpers
    ├── image_basics.py                   # resize, crop, color conversion, stats
    ├── image_processing.py               # filters, edges, threshold, contours, shapes, template
    ├── computer_vision.py                # features, face detection, YOLO object detection
    └── video_processing.py               # frame extraction, motion, tracking, video tools
```

---

## 📖 Attribution

This project is built upon the work of:

- **[GongRzhe/opencv-mcp-server](https://github.com/GongRzhe/opencv-mcp-server)** — The direct upstream project, providing the foundational OpenCV tool architecture.

---

## 📄 License

This project maintains the original repository's license. See [LICENSE](LICENSE) for details.
