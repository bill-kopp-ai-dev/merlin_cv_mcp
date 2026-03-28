"""
OpenCV MCP Server - Image Basics

This module provides basic image handling and manipulation tools using OpenCV.
It includes functionality for reading, saving, and manipulating images.
"""

import cv2
import numpy as np
import os
import logging
import datetime
import subprocess
import platform
from typing import Optional, Dict, Any, List, Union
# Import utility functions from utils
from .utils import (
    get_image_info,
    get_max_image_dimension,
    save_and_display,
    get_timestamp,
    safe_path,
    validate_int_param,
)

logger = logging.getLogger("opencv-mcp-server.image_basics")

# Tool implementations
def save_image_tool(path_in: str, path_out: str) -> Dict[str, Any]:
    """
    Salva uma cópia de uma imagem ou cria um novo arquivo a partir de outra imagem.
    
    PIPELINE E USO ESTRATÉGICO:
    - Use esta ferramenta para duplicar imagens no workspace caso necessário, ou para renomear arquivos 
      antes de processá-los com ferramentas de visão.
    
    REGRAS DE ARQUIVO:
    - 'path_in': O nome ou caminho relativo da imagem original no workspace.
    - 'path_out': O nome do novo arquivo a ser salvo (ex: 'copia_imagem.png').
    """
    path_in = str(safe_path(path_in))
    path_out = str(safe_path(path_out))
    # Read input image
    img = cv2.imread(path_in)
    if img is None:
        raise ValueError(f"Failed to read image from path: {path_in}")
    
    # Ensure directory exists
    directory = os.path.dirname(path_out)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save and display
    new_path = save_and_display(img, path_out, "save")
    
    return {
        "success": True,
        "path": new_path,
        "size_bytes": os.path.getsize(new_path),
        "size_mb": round(os.path.getsize(new_path) / (1024 * 1024), 2),
        "output_path": new_path  # Return path for chaining operations
    }

def convert_color_space_tool(image_path: str, source_space: str, target_space: str) -> Dict[str, Any]:
    """
    Converte uma imagem entre diferentes espaços de cores (ex: BGR para GRAY).
    
    PIPELINE E USO ESTRATÉGICO:
    - Use esta ferramenta como PREPARAÇÃO ANTES de chamar ferramentas de visão ou OCR, se o contraste ou cor
      forem problemas (ex: converter para 'GRAY' ajuda o LLM a ler textos borrados ou em fundos complexos).
    - Evite usar em imagens onde a cor é vital para a detecção (ex: se o usuário pede para "achar o carro vermelho").

    REGRAS DE ARQUIVO:
    - 'image_path': O nome ou caminho relativo da imagem original no workspace. A nova imagem será salva automática.
    - 'source_space': O espaço de cor atual (geralmente 'BGR').
    - 'target_space': O espaço de cor desejado (geralmente 'GRAY' ou 'RGB').
    """
    image_path = str(safe_path(image_path))
    # Read image from path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    # Map color space names to OpenCV constants
    color_space_map = {
        # "BGR" não possui constante de conversão-identidade no OpenCV;
        # o caso BGR→BGR é tratado como short-circuit antes do cvtColor.
        "RGB": cv2.COLOR_BGR2RGB,
        "GRAY": cv2.COLOR_BGR2GRAY,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "YCrCb": cv2.COLOR_BGR2YCrCb,
        "XYZ": cv2.COLOR_BGR2XYZ,
        "HSL": cv2.COLOR_BGR2HLS,
        "YUV": cv2.COLOR_BGR2YUV,
    }
    
    # Short-circuit: BGR → BGR é uma operação identidade, sem conversão.
    if source_space == "BGR" and target_space == "BGR":
        new_path = save_and_display(img, image_path, "convert_BGR_to_BGR")
        return {
            "source_space": source_space,
            "target_space": target_space,
            "info": get_image_info(img),
            "path": new_path,
            "output_path": new_path
        }
    
    # Create reverse mappings
    # Map from our naming convention to OpenCV's actual constant names
    opencv_name_map = {
        "RGB": "RGB",
        "GRAY": "GRAY",
        "HSV": "HSV",
        "LAB": "LAB",
        "YCrCb": "YCrCb",
        "XYZ": "XYZ",
        "HSL": "HLS",  # OpenCV uses HLS, not HSL
        "YUV": "YUV",
    }
    for src, code in list(color_space_map.items()):
        opencv_name = opencv_name_map.get(src, src)
        color_space_map[src + "_BGR"] = getattr(cv2, f"COLOR_{opencv_name}2BGR")
    
    # Determine the conversion code
    # Direct conversion from BGR to target
    if source_space == "BGR" and target_space in color_space_map:
        conversion_code = color_space_map[target_space]
    # Direct conversion from source to BGR
    elif target_space == "BGR" and f"{source_space}_BGR" in color_space_map:
        conversion_code = color_space_map[f"{source_space}_BGR"]
    # Check for explicit conversion key
    elif f"{source_space}_{target_space}" in color_space_map:
        conversion_code = color_space_map[f"{source_space}_{target_space}"]
    else:
        # Handle two-step conversion via BGR
        src_to_bgr_key = f"{source_space}_BGR"
        bgr_to_tgt_key = target_space  # BGR to target is just the target name
        
        if src_to_bgr_key in color_space_map and bgr_to_tgt_key in color_space_map:
            # First convert to BGR
            img = cv2.cvtColor(img, color_space_map[src_to_bgr_key])
            # Then convert from BGR to target
            conversion_code = color_space_map[bgr_to_tgt_key]
        else:
            raise ValueError(f"Unsupported color space conversion: {source_space} to {target_space}")
    
    # Perform the conversion
    converted = cv2.cvtColor(img, conversion_code)
    
    # For display purposes, convert back to BGR if not already in BGR
    display_img = converted
    if target_space != "BGR" and f"{target_space}_BGR" in color_space_map:
        display_img = cv2.cvtColor(converted, color_space_map[f"{target_space}_BGR"])
    
    # Save and display
    new_path = save_and_display(display_img, image_path, f"convert_{source_space}_to_{target_space}")
    
    return {
        "source_space": source_space,
        "target_space": target_space,
        "info": get_image_info(converted),
        "path": new_path,
        "output_path": new_path  # Return path for chaining operations
    }

def resize_image_tool(image_path: str, width: int, height: int, interpolation: str = "INTER_LINEAR") -> Dict[str, Any]:
    """
    Redimensiona uma imagem para uma largura e altura exatas em pixels.
    
    PIPELINE E USO ESTRATÉGICO:
    - Use esta ferramenta OBRIGATORIAMENTE se a imagem original for excessivamente grande 
      (ex: maior que 2000x2000) ANTES de enviá-la para um modelo de visão, para evitar erros de 'Payload Too Large'.
    - Use esta ferramenta para ajustar imagens geradas caso o usuário peça um tamanho específico.
    
    REGRAS DE ARQUIVO:
    - 'image_path': O nome ou caminho relativo da imagem de origem no workspace.
    - 'width' e 'height': As novas dimensões. A imagem será salva automaticamente.
    - 'interpolation': Método de interpolação (padrão MANTIDO se não especificado).

    LIMITES E SEGURANÇA:
    - `width` e `height` são validados como inteiros positivos.
    - O limite superior é controlado por `MERLIN_CV_MAX_IMAGE_DIMENSION` para evitar uso excessivo.

    RETORNO RELEVANTE PARA ORQUESTRAÇÃO:
    - `output_path`: caminho final para encadear com outras tools.
    - `info` e `original_info`: úteis para decidir se nova redução ainda é necessária.
    """
    image_path = str(safe_path(image_path))
    max_dimension = get_max_image_dimension()
    width = validate_int_param("width", width, minimum=1, maximum=max_dimension)
    height = validate_int_param("height", height, minimum=1, maximum=max_dimension)
    # Read image from path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    # Map interpolation methods to OpenCV constants
    interp_methods = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4
    }
    
    interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)
    
    # Perform the resize
    resized = cv2.resize(img, (width, height), interpolation=interp)
    
    # Save and display
    new_path = save_and_display(resized, image_path, f"resize_{width}x{height}")
    
    return {
        "width": width,
        "height": height,
        "interpolation": interpolation,
        "info": get_image_info(resized),
        "original_info": get_image_info(img),
        "path": new_path,
        "output_path": new_path  # Return path for chaining operations
    }

def crop_image_tool(image_path: str, x: int, y: int, width: int, height: int) -> Dict[str, Any]:
    """
    Recorta uma região específica de uma imagem e salva como um novo arquivo.
    
    PIPELINE E USO ESTRATÉGICO:
    - Use esta ferramenta ANTES de usar ferramentas de visão (como identificar objetos ou ler texto) 
      se o usuário pedir para analisar um detalhe específico, um canto ou um objeto pequeno em uma imagem maior.
    - Isso reduz o ruído e melhora drasticamente a precisão da análise do LLM de Visão.
    
    REGRAS DE ARQUIVO:
    - 'image_path': O nome ou caminho relativo da imagem de origem no workspace.
    - As coordenadas x, y denotam o canto superior esquerdo do recorte. width e height indicam o tamanho.
    - O recorte será salvo automaticamente com um sufixo no workspace.
    """
    image_path = str(safe_path(image_path))
    x = validate_int_param("x", x, minimum=0)
    y = validate_int_param("y", y, minimum=0)
    width = validate_int_param("width", width, minimum=1)
    height = validate_int_param("height", height, minimum=1)
    # Read image from path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Validate crop parameters
    if x < 0 or y < 0 or (x + width) > img_width or (y + height) > img_height:
        raise ValueError(f"Crop region ({x}, {y}, {width}, {height}) is outside image bounds ({img_width}, {img_height})")
    
    # Perform the crop
    cropped = img[y:y+height, x:x+width]
    
    # Save and display
    new_path = save_and_display(cropped, image_path, f"crop_{x}_{y}_{width}_{height}")
    
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "info": get_image_info(cropped),
        "original_info": get_image_info(img),
        "path": new_path,
        "output_path": new_path  # Return path for chaining operations
    }

def get_image_stats_tool(image_path: str, channels: bool = True) -> Dict[str, Any]:
    """
    Calcula e retorna informações estatísticas de uma imagem (histograma, média, desvio padrão).
    
    PIPELINE E USO ESTRATÉGICO:
    - Raramente usado antes de análise visual de LLM, a menos que o usuário exija dados numéricos precisos
      sobre contraste, brilho global ou balanço de cores da imagem.
      
    REGRAS DE ARQUIVO:
    - 'image_path': O nome ou caminho relativo da imagem no workspace.
    - 'channels': Booleano indicando cálculo de cor individual.
    """
    image_path = str(safe_path(image_path))
    # Read image from path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    # Calculate basic statistics
    info = get_image_info(img)
    
    # Calculate mean and standard deviation
    mean, stddev = cv2.meanStdDev(img)
    
    stats = {
        "info": info,
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "stddev": float(np.std(img))
    }
    
    # Calculate per-channel statistics if requested
    if channels and len(img.shape) > 2:
        channel_stats = []
        for i in range(img.shape[2]):
            channel = img[:,:,i]
            channel_stats.append({
                "channel": i,
                "min": float(np.min(channel)),
                "max": float(np.max(channel)),
                "mean": float(np.mean(channel)),
                "stddev": float(np.std(channel))
            })
        stats["channels"] = channel_stats
    
    # Calculate histogram data
    hist_data = []
    if len(img.shape) > 2:  # Color image
        for i in range(img.shape[2]):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist_data.append(hist.flatten().tolist())
    else:  # Grayscale image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_data.append(hist.flatten().tolist())
    
    stats["histogram"] = hist_data
    
    # Create a visual histogram
    hist_img = np.zeros((400, 512, 3), np.uint8)
    hist_img[:] = (255, 255, 255)  # White background
    
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # BGR for each channel
    
    for i, hist_channel in enumerate(hist_data):
        if i >= 3:  # Only handle up to 3 channels
            break
            
        # Normalize histogram for display
        hist_max = max(hist_channel)
        if hist_max > 0:
            normalized = [h * 380 / hist_max for h in hist_channel]
            
            # Draw histogram lines
            for x in range(256):
                pt1 = (2*x, 399)
                pt2 = (2*x, 399 - int(normalized[x]))
                cv2.line(hist_img, pt1, pt2, colors[i], 1)
    
    # Save and display histogram
    hist_path = save_and_display(hist_img, image_path, "histogram")
    
    # Display the original image as well
    display_path = save_and_display(img, image_path, "stats")
    
    stats["histogram_image_path"] = hist_path
    stats["image_path"] = display_path
    stats["output_path"] = display_path  # Return path for chaining operations
    
    return stats

def register_tools(mcp):
    """
    Register all image basics tools with the MCP server
    
    Args:
        mcp: The MCP server instance
    """
    # Register optimized tool implementations
    mcp.add_tool(save_image_tool)
    mcp.add_tool(convert_color_space_tool)
    mcp.add_tool(resize_image_tool)
    mcp.add_tool(crop_image_tool)
    mcp.add_tool(get_image_stats_tool)
