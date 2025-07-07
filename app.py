# app.py

import streamlit as st
import os
import json
import zipfile
import tempfile
import subprocess
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytesseract
import difflib
import html
import time
from pathlib import Path
import platform
import shutil
from copy import deepcopy
import math
import logging
from PIL import Image
import io

# Try to import additional OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Page config
st.set_page_config(page_title="Document Number Processor", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Utility functions ----------------

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_text_from_docx(docx_path: str) -> str:
    """Extract all text from DOCX: document.xml, headers, footers."""
    text_chunks = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(docx_path, 'r') as zin:
                zin.extractall(tmpdir)
            xml_paths = ['word/document.xml'] + \
                        [f"word/header{i}.xml" for i in range(1, 4)] + \
                        [f"word/footer{i}.xml" for i in range(1, 4)]
            for rel in xml_paths:
                xml_file = tmpdir_path / rel
                if not xml_file.exists():
                    continue
                try:
                    tree = ET.parse(str(xml_file))
                    root = tree.getroot()
                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    for t in root.findall('.//w:t', ns):
                        if t.text:
                            text_chunks.append(t.text)
                except Exception:
                    continue
    except Exception:
        return ""
    return " ".join(text_chunks)

def extract_images_from_docx(docx_path: str):
    """
    Return list of image bytes from DOCX's word/media folder.
    """
    images = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(docx_path, 'r') as zin:
                zin.extractall(tmpdir)
            media_dir = tmpdir_path / 'word' / 'media'
            if media_dir.exists():
                for img_file in media_dir.iterdir():
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                        try:
                            images.append(img_file.read_bytes())
                        except Exception:
                            pass
    except Exception:
        pass
    return images

def highlight_differences(text1: str, text2: str) -> str:
    """Return HTML highlighting word-level additions/removals between text1 and text2."""
    diff = difflib.ndiff(text1.split(), text2.split())
    out = []
    for token in diff:
        code = token[:2]
        word = html.escape(token[2:])
        if code == '  ':
            out.append(word)
        elif code == '- ':
            out.append(f"<span style='background:#ffe6e6; text-decoration:line-through'>{word}</span>")
        elif code == '+ ':
            out.append(f"<span style='background:#e6ffe6'>{word}</span>")
    return " ".join(out)

def enhance_image_for_ocr(image):
    """Apply advanced preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small
    h, w = gray.shape
    if max(h, w) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply multiple preprocessing techniques
    # 1. Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11, 7
    )
    
    thresh2 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 7
    )
    
    # 2. Noise reduction
    denoised1 = cv2.fastNlMeansDenoising(thresh1, None, 10, 7, 21)
    denoised2 = cv2.fastNlMeansDenoising(thresh2, None, 10, 7, 21)
    
    # 3. Edge-preserving smoothing
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Combine results
    combined = cv2.addWeighted(denoised1, 0.4, denoised2, 0.4, 0)
    combined = cv2.addWeighted(combined, 0.7, sharpened, 0.3, 0)
    
    # Final thresholding
    _, final = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return final

def wrap_text(text, font, font_scale, thickness, max_width):
    """Wrap text to fit within max_width"""
    lines = []
    words = text.split()
    
    if not words:
        return [""]
    
    current_line = words[0]
    
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    lines.append(current_line)
    return lines

def calculate_font_scale(text, width, height, max_font_scale=1.5, min_font_scale=0.3):
    """Calculate optimal font scale to fit text within boundaries"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    padding = 5
    
    # Start with max font scale
    font_scale = max_font_scale
    while font_scale >= min_font_scale:
        # Wrap text to fit width
        lines = wrap_text(text, font, font_scale, thickness, width - 2 * padding)
        
        # Calculate total height needed
        total_height = 0
        max_line_width = 0
        
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            total_height += h + 5  # Add line spacing
            if w > max_line_width:
                max_line_width = w
        
        # Check if it fits
        if total_height <= height - 2 * padding and max_line_width <= width - 2 * padding:
            return font_scale, lines
        
        # Reduce font scale and try again
        font_scale -= 0.05
    
    # If we reach here, use min font scale and single line
    return min_font_scale, [text]

def draw_text_within_bounds(image, text, x, y, width, height, color=(0, 0, 0)):
    """Draw text within specified boundaries with auto scaling and wrapping"""
    # Cover original text area
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), -1)
    
    # Calculate optimal font scale and wrap text
    font_scale, lines = calculate_font_scale(text, width, height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))
    
    # Calculate starting y position
    total_text_height = 0
    for line in lines:
        (_, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        total_text_height += h + 5
    
    start_y = y + (height - total_text_height) // 2
    
    # Draw each line
    for i, line in enumerate(lines):
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        pos_x = x + (width - w) // 2
        pos_y = start_y + h
        
        cv2.putText(
            image,
            line,
            (pos_x, pos_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        start_y += h + 5  # Move to next line

def tesseract_ocr(image, languages, whitelist):
    """Perform OCR using Tesseract"""
    # Preprocessing
    preprocessed = enhance_image_for_ocr(image)
    
    # OCR configuration
    config = f"--psm 6 --oem 3 -l {languages} -c tessedit_char_whitelist={whitelist}"
    
    try:
        data = pytesseract.image_to_data(
            preprocessed, 
            output_type=pytesseract.Output.DICT, 
            config=config
        )
        return data
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}


def paddle_ocr(image, languages):
    """Perform OCR using PaddleOCR if available"""
    if not PADDLEOCR_AVAILABLE:
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
    
    try:
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize PaddleOCR with updated API
        try:
            # Try with new parameter name
            ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
        except TypeError:
            # Fallback to old parameter name if new one doesn't work
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Perform OCR with updated API
        result = ocr.ocr(image_rgb)
        
        # Process results
        texts = []
        lefts = []
        tops = []
        widths = []
        heights = []
        confs = []
        
        if result is None:
            return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
            
        for line in result:
            if line is None:
                continue
            for item in line:
                if item is None:
                    continue
                points = item[0]
                text, conf = item[1]
                if not text.strip():
                    continue
                
                # Extract bounding box coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y
                
                texts.append(text)
                lefts.append(int(x))
                tops.append(int(y))
                widths.append(int(w))
                heights.append(int(h))
                confs.append(conf * 100)  # Convert to percentage
        
        return {
            'text': texts,
            'left': lefts,
            'top': tops,
            'width': widths,
            'height': heights,
            'conf': confs
        }
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}

def easy_ocr(image, languages):
    """Perform OCR using EasyOCR if available"""
    if not EASYOCR_AVAILABLE:
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
    
    try:
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])
        
        # Perform OCR
        result = reader.readtext(image_rgb)
        
        # Process results
        texts = []
        lefts = []
        tops = []
        widths = []
        heights = []
        confs = []
        
        for item in result:
            points, text, conf = item
            if not text.strip():
                continue
            
            # Extract bounding box coordinates
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            
            texts.append(text)
            lefts.append(int(x))
            tops.append(int(y))
            widths.append(int(w))
            heights.append(int(h))
            confs.append(conf * 100)  # Convert to percentage
        
        return {
            'text': texts,
            'left': lefts,
            'top': tops,
            'width': widths,
            'height': heights,
            'conf': confs
        }
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}

def combine_ocr_results(tesseract_data, paddle_data, easy_data):
    """Combine results from multiple OCR engines, prioritizing Tesseract"""
    combined = {
        'text': [],
        'left': [],
        'top': [],
        'width': [],
        'height': [],
        'conf': []
    }
    
    # Add Tesseract results
    combined['text'].extend(tesseract_data['text'])
    combined['left'].extend(tesseract_data['left'])
    combined['top'].extend(tesseract_data['top'])
    combined['width'].extend(tesseract_data['width'])
    combined['height'].extend(tesseract_data['height'])
    combined['conf'].extend(tesseract_data['conf'])
    
    # Add PaddleOCR results that don't overlap significantly with Tesseract
    for i in range(len(paddle_data['text'])):
        paddle_rect = (paddle_data['left'][i], paddle_data['top'][i], 
                      paddle_data['width'][i], paddle_data['height'][i])
        overlap = False
        
        for j in range(len(combined['text'])):
            tesseract_rect = (combined['left'][j], combined['top'][j],
                             combined['width'][j], combined['height'][j])
            
            # Check for significant overlap
            if has_significant_overlap(paddle_rect, tesseract_rect):
                overlap = True
                break
        
        if not overlap:
            combined['text'].append(paddle_data['text'][i])
            combined['left'].append(paddle_data['left'][i])
            combined['top'].append(paddle_data['top'][i])
            combined['width'].append(paddle_data['width'][i])
            combined['height'].append(paddle_data['height'][i])
            combined['conf'].append(paddle_data['conf'][i])
    
    # Add EasyOCR results that don't overlap significantly with existing results
    for i in range(len(easy_data['text'])):
        easy_rect = (easy_data['left'][i], easy_data['top'][i], 
                    easy_data['width'][i], easy_data['height'][i])
        overlap = False
        
        for j in range(len(combined['text'])):
            existing_rect = (combined['left'][j], combined['top'][j],
                            combined['width'][j], combined['height'][j])
            
            # Check for significant overlap
            if has_significant_overlap(easy_rect, existing_rect):
                overlap = True
                break
        
        if not overlap:
            combined['text'].append(easy_data['text'][i])
            combined['left'].append(easy_data['left'][i])
            combined['top'].append(easy_data['top'][i])
            combined['width'].append(easy_data['width'][i])
            combined['height'].append(easy_data['height'][i])
            combined['conf'].append(easy_data['conf'][i])
    
    return combined

def has_significant_overlap(rect1, rect2, threshold=0.5):
    """Check if two rectangles have significant overlap"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calculate intersection area
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Check if intersection area is significant relative to either rectangle
    return (intersection_area / min(area1, area2)) > threshold

def process_image_bytes(image_bytes: bytes, languages: str, compiled_patterns: list, use_multiple_ocr: bool):
    """
    Enhanced OCR with multiple engines and robust pattern detection
    Return tuple: (original_bytes, processed_bytes, matches_found, replacements_done)
    """
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            return image_bytes, image_bytes, 0, 0
        
        # Create a copy for processing
        processed_img = original_img.copy()
        
        # Character whitelist
        whitelist_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
        
        # Perform OCR with multiple engines
        tesseract_data = tesseract_ocr(original_img, languages, whitelist_chars)
        paddle_data = paddle_ocr(original_img, languages) if use_multiple_ocr else {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
        easy_data = easy_ocr(original_img, languages) if use_multiple_ocr else {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
        
        # Combine results
        if use_multiple_ocr:
            data = combine_ocr_results(tesseract_data, paddle_data, easy_data)
        else:
            data = tesseract_data
        
        matches_found = 0
        replacements_done = 0
        
        # Process each detected text element
        n = len(data.get('text', []))
        for i in range(n):
            text = data['text'][i].strip()
            conf = data.get('conf', [0]*n)[i]
            
            # Skip empty text and low confidence detections
            if not text or conf < 60:
                continue
            
            # Get bounding box coordinates
            try:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            except Exception:
                continue
            
            # Check for matches with patterns
            for regex, repl in compiled_patterns:
                # Find all matches in this text segment
                found_matches = list(regex.finditer(text))
                if found_matches:
                    for match in found_matches:
                        # Log detected pattern for debugging
                        logger.info(f"Detected pattern: {match.group(0)}")
                        matches_found += 1
                    
                    # Apply replacements to each match
                    new_text = text
                    for match in found_matches:
                        # Apply replacement
                        if callable(repl):
                            try:
                                rep_str = repl(match)
                            except Exception:
                                rep_str = match.group(0)
                        else:
                            try:
                                rep_str = regex.sub(repl, match.group(0))
                            except re.error:
                                rep_str = match.group(0)
                        
                        # Only count as replacement if text changed
                        if rep_str != match.group(0):
                            replacements_done += 1
                        
                        # Replace the matched portion
                        new_text = new_text.replace(match.group(0), rep_str, 1)
                    
                    # Only update if changes were made
                    if new_text != text:
                        # Draw text within original bounds
                        draw_text_within_bounds(processed_img, new_text, x, y, w, h)
        
        # Convert processed image back to bytes
        success, processed_bytes = cv2.imencode('.png', processed_img)
        if not success:
            return image_bytes, image_bytes, matches_found, replacements_done
            
        return image_bytes, processed_bytes.tobytes(), matches_found, replacements_done
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return image_bytes, image_bytes, 0, 0



def insert_image_after_original(doc, img_path, rel_id, new_img_path, new_rel_id):
    """
    Enhanced to handle images in tables and complex layouts
    """
    W_NAMESPACE = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    WP_NAMESPACE = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    A_NAMESPACE = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    ns = {
        'w': W_NAMESPACE,
        'wp': WP_NAMESPACE,
        'a': A_NAMESPACE
    }
    
    # Find all drawings containing the original image
    drawings = []
    for drawing in doc.findall('.//w:drawing', ns):
        blip = drawing.find('.//a:blip', ns)
        if blip is not None and blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed') == rel_id:
            drawings.append(drawing)
    
    # Process each matching drawing
    for drawing in drawings:
        # Find parent run
        parent = next((e for e in doc.iter() if drawing in list(e)), None)
        if parent is None:
            continue
            
        # Find grandparent (paragraph or table cell)
        grandparent = next((e for e in doc.iter() if parent in list(e)), None)
        if grandparent is None:
            continue
            
        # Create a new paragraph with the new image
        new_p = ET.Element(f"{{{W_NAMESPACE}}}p")
        
        # Create a new run and copy the drawing element
        new_r = ET.Element(f"{{{W_NAMESPACE}}}r")
        new_drawing = deepcopy(drawing)
        
        # Update the relationship ID in the copied drawing
        new_blip = new_drawing.find('.//a:blip', ns)
        if new_blip is not None:
            new_blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed', new_rel_id)
        
        new_r.append(new_drawing)
        new_p.append(new_r)
        
        # Insert after the current element
        if grandparent.tag == f'{{{W_NAMESPACE}}}p':
            # Find parent of grandparent (usually body or table cell)
            great_grandparent = next((e for e in doc.iter() if grandparent in list(e)), None)
            if great_grandparent is not None:
                index = list(great_grandparent).index(grandparent)
                great_grandparent.insert(index + 1, new_p)
                return True
        elif grandparent.tag == f'{{{W_NAMESPACE}}}tc':
            # Insert directly into table cell
            grandparent.append(new_p)
            return True
            
    return False


def replace_patterns_in_docx(
    input_path: str,
    output_path: str,
    patterns: list,
    ocr_langs: str,
    include_images: bool,
    use_multiple_ocr: bool,
    progress_callback=None
):
    """
    Process DOCX file with enhanced image handling:
    - Text: append replacements after original text
    - Images: create new image with replacements and insert below original
    """
    # Compile patterns
    compiled = []
    for p in patterns:
        pat = p.get("pattern")
        repl = p.get("replacement")
        if not pat:
            continue
        try:
            regex = re.compile(pat, re.IGNORECASE)
        except re.error:
            continue
        if isinstance(repl, str):
            compiled.append((regex, repl))
        elif callable(repl):
            compiled.append((regex, repl))
        else:
            continue

    # Counters
    text_parts = 0
    text_matches = 0
    text_replacements = 0
    total_media = 0
    images_processed = 0
    image_matches = 0
    image_replacements = 0

    # Namespaces
    W_NAMESPACE = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    ns = {'w': W_NAMESPACE}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Unzip DOCX
        with zipfile.ZipFile(input_path, 'r') as zin:
            zin.extractall(tmpdir)

        # Identify XML parts to process
        xml_rel_paths = ['word/document.xml'] + \
                        [f"word/header{i}.xml" for i in range(1, 4)] + \
                        [f"word/footer{i}.xml" for i in range(1, 4)]
        existing_xml = [rel for rel in xml_rel_paths if (tmpdir_path / rel).exists()]
        text_parts = len(existing_xml)

        # Count images
        media_dir = tmpdir_path / 'word' / 'media'
        if include_images and media_dir.exists():
            total_media = sum(1 for f in media_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp'])
        # Total steps for progress
        total_steps = max(1, text_parts + total_media)
        step = 0

        # Process text parts
        for rel in existing_xml:
            xml_file = tmpdir_path / rel
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()

                # First, collect paragraphs inside textboxes:
                textbox_paras = set(root.findall('.//w:txbxContent//w:p', ns))
                # All paragraphs:
                all_paras = root.findall('.//w:p', ns)

                for p in all_paras:
                    in_textbox = p in textbox_paras
                    # List of runs
                    runs = list(p.findall('w:r', ns))
                    for idx, r in enumerate(runs):
                        t = r.find('w:t', ns)
                        if t is not None and t.text:
                            orig_text = t.text
                            appended_texts = []
                            # For each pattern, find matches
                            for regex, repl in compiled:
                                try:
                                    matches = list(regex.finditer(orig_text))
                                except re.error:
                                    matches = []
                                if matches:
                                    for m in matches:
                                        # Build replacement text
                                        if callable(repl):
                                            try:
                                                rep = repl(m)
                                            except Exception:
                                                rep = m.group(0)
                                        else:
                                            try:
                                                rep = regex.sub(repl, m.group(0))
                                            except re.error:
                                                rep = m.group(0)
                                        appended_texts.append(rep)
                                        text_matches += 1
                                        text_replacements += 1
                            if appended_texts:
                                # Combine appended texts separated by commas, prefix comma
                                combined = "," + ",".join(appended_texts)
                                # Create new run <w:r>
                                new_r = ET.Element(f"{{{W_NAMESPACE}}}r")
                                if in_textbox:
                                    # Copy run properties so formatting/wrapping preserved
                                    orig_rpr = r.find('w:rPr', ns)
                                    if orig_rpr is not None:
                                        new_rpr = deepcopy(orig_rpr)
                                        new_r.append(new_rpr)
                                # Otherwise (not in textbox), do not copy run properties
                                new_t = ET.SubElement(new_r, f"{{{W_NAMESPACE}}}t")
                                new_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                                new_t.text = combined
                                # Insert after current run
                                children = list(p)
                                insert_idx = None
                                for i_ch, ch in enumerate(children):
                                    if ch is r:
                                        insert_idx = i_ch + 1
                                        break
                                if insert_idx is not None:
                                    p.insert(insert_idx, new_r)
                                else:
                                    p.append(new_r)
                # Save XML
                tree.write(str(xml_file), encoding='utf-8', xml_declaration=True)
            except Exception as e:
                logger.error(f"Error processing text part {rel}: {e}")
            step += 1
            if progress_callback:
                percent = int(round(100 * step / total_steps))
                percent = max(0, min(100, percent))
                summary = {
                    'text_parts': text_parts,
                    'text_matches_found': text_matches,
                    'text_replacements_done': text_replacements,
                    'total_media': total_media,
                    'images_processed': images_processed,
                    'image_matches_found': image_matches,
                    'image_replacements_done': image_replacements
                }
                progress_callback(percent, summary)

        # Process images and insert new versions below originals
        if include_images and media_dir.exists():
            # Build relationship maps for each document part
            rels_maps = {}
            for rel in existing_xml:
                rels_file = tmpdir_path / 'word' / '_rels' / (Path(rel).name + '.rels')
                if rels_file.exists():
                    try:
                        rels_tree = ET.parse(str(rels_file))
                        rels_root = rels_tree.getroot()
                        rels_map = {}
                        for relationship in rels_root.findall('Relationship', {'': 'http://schemas.openxmlformats.org/package/2006/relationships'}):
                            if 'image' in relationship.get('Type', ''):
                                rels_map[relationship.get('Id')] = relationship.get('Target')
                        rels_maps[str(rels_file)] = rels_map
                    except Exception as e:
                        logger.error(f"Error reading relationships for {rel}: {e}")
                        rels_maps[str(rels_file)] = {}
            
            # Process each image
            for img_file in media_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        img_bytes = img_file.read_bytes()
                        orig_bytes, processed_bytes, im_matches, im_repls = process_image_bytes(
                            img_bytes, ocr_langs, compiled, use_multiple_ocr
                        )
                        
                        images_processed += 1
                        image_matches += im_matches
                        image_replacements += im_repls
                        
                        # Only create new image if replacements were done
                        if im_repls > 0:
                            # Save processed image
                            new_img_name = f"processed_{img_file.name}"
                            new_img_path = media_dir / new_img_name
                            with open(new_img_path, "wb") as f:
                                f.write(processed_bytes)
                            
                            # Find original image in all document parts
                            for rel in existing_xml:
                                xml_file = tmpdir_path / rel
                                rels_file = tmpdir_path / 'word' / '_rels' / (Path(rel).name + '.rels')
                                rels_map = rels_maps.get(str(rels_file), {})
                                
                                # Find relationship ID for original image
                                orig_rel_id = None
                                for rid, target in rels_map.items():
                                    if target.endswith(img_file.name):
                                        orig_rel_id = rid
                                        break
                                
                                if orig_rel_id:
                                    try:
                                        # Add new relationship
                                        rels_tree = ET.parse(str(rels_file))
                                        rels_root = rels_tree.getroot()
                                        
                                        # Create new relationship ID
                                        new_rel_id = f"rId{len(rels_root) + 1000}"
                                        ns_rel = {'': 'http://schemas.openxmlformats.org/package/2006/relationships'}
                                        ET.SubElement(
                                            rels_root, 
                                            'Relationship', 
                                            {
                                                'Id': new_rel_id,
                                                'Type': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image',
                                                'Target': f'media/{new_img_name}'
                                            }
                                        )
                                        
                                        # Save updated relationships
                                        rels_tree.write(str(rels_file), encoding='utf-8', xml_declaration=True)
                                        
                                        # Update XML to insert new image after original
                                        tree = ET.parse(str(xml_file))
                                        root = tree.getroot()
                                        inserted = insert_image_after_original(root, img_file.name, orig_rel_id, new_img_name, new_rel_id)
                                        if inserted:
                                            tree.write(str(xml_file), encoding='utf-8', xml_declaration=True)
                                        else:
                                            logger.warning(f"Failed to insert new image for {img_file.name} in {rel}")
                                            
                                    except Exception as e:
                                        logger.error(f"Error updating relationships for {img_file.name}: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_file.name}: {e}")
                    step += 1
                    if progress_callback:
                        percent = int(round(100 * step / total_steps))
                        percent = max(0, min(100, percent))
                        summary = {
                            'text_parts': text_parts,
                            'text_matches_found': text_matches,
                            'text_replacements_done': text_replacements,
                            'total_media': total_media,
                            'images_processed': images_processed,
                            'image_matches_found': image_matches,
                            'image_replacements_done': image_replacements
                        }
                        progress_callback(percent, summary)

        # Repack DOCX
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for folder, _, files in os.walk(tmpdir):
                for fname in files:
                    full = Path(folder) / fname
                    arc = full.relative_to(tmpdir_path)
                    zout.write(str(full), str(arc))

    return {
        'text_parts': text_parts,
        'text_matches_found': text_matches,
        'text_replacements_done': text_replacements,
        'total_media': total_media,
        'images_processed': images_processed,
        'image_matches_found': image_matches,
        'image_replacements_done': image_replacements
    }

def load_patterns():
    """Sidebar UI: load patterns from JSON upload or manual entry."""
    st.sidebar.subheader("🔍 Pattern Configuration")
    col1, col2 = st.sidebar.columns([2, 1])
    uploaded = None
    with col1:
        uploaded = st.file_uploader("Upload patterns.json", type=["json"], key="patterns_upload")
    with col2:
        st.write("or")
    manual = st.sidebar.text_area(
        "Manual patterns (JSON list)", height=150,
        help='Example: [ {"pattern": "77-([A-Za-z0-9]{2,10})-(\\d{5,10})(?:-([A-Za-z0-9]{2,10}))?", "replacement": ""} ]\n'
             'If "replacement" is omitted or empty, default logic is used for default pattern.',
        key="patterns_manual"
    )
    patterns = []
    if uploaded:
        try:
            patterns = json.load(uploaded)
            if not isinstance(patterns, list):
                st.sidebar.error("patterns.json must be a JSON list of objects")
                patterns = []
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")
            patterns = []
    elif manual:
        try:
            patterns = json.loads(manual)
            if not isinstance(patterns, list):
                st.sidebar.error("Manual JSON must be a list of objects")
                patterns = []
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")
            patterns = []
    # Default if none or invalid
    if not patterns:
        st.sidebar.info("Using enhanced pattern: Detects all 77-XXXX-XXXXX variations")
        # Enhanced pattern to handle all observed variations
        default_pattern = r"77\s*[-–—]?\s*([A-Za-z0-9]{2,10})\s*[-–—]?\s*([A-Za-z0-9]{5,10})(?:\s*[-–—]?\s*([A-Za-z0-9]{2,10}))?"
        def default_repl(m):
            g1 = m.group(1)
            g2 = m.group(2)
            g3 = m.group(3) if m.group(3) else ""
            if g3:
                return f"4022-{g1}-{g2}-{g3}"
            else:
                return f"4022-{g1}-{g2}"
        patterns = [
            {
                "pattern": default_pattern,
                "replacement": default_repl
            }
        ]
    return patterns

def attempt_doc_to_docx(input_path: str) -> str:
    """
    If input_path ends with .doc, attempt conversion to .docx via soffice/libreoffice.
    Return path to converted .docx, or raise Exception if fails.
    """
    ext = Path(input_path).suffix.lower()
    if ext != ".doc":
        return input_path
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise FileNotFoundError("LibreOffice/soffice not found for .doc to .docx conversion.")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(input_path).stem
        tmp_doc = Path(tmpdir) / f"{base}.doc"
        shutil.copy(input_path, tmp_doc)
        cmd = [soffice, "--headless", "--convert-to", "docx", "--outdir", tmpdir, str(tmp_doc)]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Conversion failed: {e.stderr.decode(errors='ignore')}")
        converted = Path(tmpdir) / f"{base}.docx"
        if converted.exists():
            dest = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
            dest.close()
            shutil.copy(str(converted), dest.name)
            return dest.name
        else:
            raise FileNotFoundError("Converted .docx not found after conversion.")

def check_tesseract():
    """Check Tesseract availability; return version string or None."""
    try:
        version = pytesseract.get_tesseract_version()
        return str(version)
    except Exception:
        return None

def match_files(original_files, processed_files):
    """
    Match UploadedFile lists by filename (stem).
    Returns:
      matched: list of tuples (orig_file, proc_file)
      unmatched_originals: list of orig_file not matched
      unmatched_processed: list of proc_file not matched

    Logic:
    - Build a map from processed stem to processed_file
      If processed filename stem starts with 'processed_', strip that prefix.
    - For each original, match by stem in that map.
    """
    matched = []
    unmatched_o = []
    unmatched_p = []

    # Build map: stem -> processed_file
    proc_map = {}
    for pf in processed_files:
        stem = Path(pf.name).stem
        if stem.startswith("processed_"):
            orig_stem = stem[len("processed_"):]
            proc_map[orig_stem] = pf
        else:
            proc_map[stem] = pf

    # Match originals
    for of in original_files:
        o_stem = Path(of.name).stem
        if o_stem in proc_map:
            matched.append((of, proc_map[o_stem]))
        else:
            unmatched_o.append(of)

    # Determine unmatched processed by name
    matched_proc_names = {p.name for _, p in matched}
    for pf in processed_files:
        if pf.name not in matched_proc_names:
            unmatched_p.append(pf)

    return matched, unmatched_o, unmatched_p

def compare_document_pair(orig_file, proc_file):
    """
    orig_file, proc_file: UploadedFile-like with .name and .getbuffer()
    Returns dict with keys: original, processed, text_diff_html, orig_images, proc_images, error
    """
    orig_path = proc_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(orig_file.getbuffer())
            orig_path = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f2:
            f2.write(proc_file.getbuffer())
            proc_path = f2.name

        orig_text = extract_text_from_docx(orig_path)
        proc_text = extract_text_from_docx(proc_path)
        text_diff_html = highlight_differences(orig_text, proc_text)

        orig_images = extract_images_from_docx(orig_path)
        proc_images = extract_images_from_docx(proc_path)

        return {
            'original': orig_file.name,
            'processed': proc_file.name,
            'text_diff_html': text_diff_html,
            'orig_images': orig_images,
            'proc_images': proc_images,
            'error': None
        }
    except Exception as e:
        return {
            'original': getattr(orig_file, 'name', ""),
            'processed': getattr(proc_file, 'name', ""),
            'text_diff_html': "",
            'orig_images': [],
            'proc_images': [],
            'error': str(e)
        }
    finally:
        for p in (orig_path, proc_path):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

# ---------------- Main UI Pages ----------------

def single_file_page(patterns, ocr_langs, include_images, use_multiple_ocr):
    st.header("Single Document Processing")
    uploaded = st.file_uploader("Upload a .doc or .docx file", type=["doc", "docx"], key="single_upload")
    if not uploaded:
        st.info("Please upload a file to proceed.")
        return
    suffix = Path(uploaded.name).suffix.lower()
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(uploaded.getbuffer())
    tmp_in_path = tmp_in.name
    tmp_in.close()

    converted_path = tmp_in_path
    if suffix == ".doc":
        st.info("Attempting .doc → .docx conversion...")
        try:
            converted = attempt_doc_to_docx(tmp_in_path)
            converted_path = converted
            st.success("Conversion succeeded.")
        except Exception as e:
            st.error(f".doc conversion failed: {e}")
            return

    with st.expander("📝 Original Text Preview", expanded=False):
        try:
            orig_text = extract_text_from_docx(converted_path)
        except Exception as e:
            orig_text = f"Error: {e}"
        st.text_area("Original text", orig_text, height=200)

    if st.button("▶️ Run Replacement (Single)"):
        progress_bar = st.progress(0)
        status = st.empty()
        metrics_area = st.empty()
        start_time = time.time()

        output_dir = st.text_input("Output directory on server", value="output_docs")
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"processed_{Path(uploaded.name).stem}.docx"
        out_path = os.path.join(output_dir, out_name)

        status.text("🔄 Processing...")
        # Live callback
        def callback(percent, summary):
            try:
                progress_bar.progress(percent)
            except Exception:
                pass
            elapsed = time.time() - start_time
            tm = summary.get('text_matches_found', 0)
            tr = summary.get('text_replacements_done', 0)
            im_p = summary.get('images_processed', 0)
            tot_m = summary.get('total_media', 0)
            im_m = summary.get('image_matches_found', 0)
            im_r = summary.get('image_replacements_done', 0)
            metrics_area.markdown(
                f"Elapsed: {format_time(elapsed)} | Text parts: {summary.get('text_parts',0)} | "
                f"Text matches: {tm} | Text appended: {tr} | "
                f"Images processed: {im_p}/{tot_m} | Image matches: {im_m} | Image replacements: {im_r}"
            )

        try:
            summary = replace_patterns_in_docx(
                input_path=converted_path,
                output_path=out_path,
                patterns=patterns,
                ocr_langs=ocr_langs,
                include_images=include_images,
                use_multiple_ocr=use_multiple_ocr,
                progress_callback=callback
            )
        except Exception as e:
            st.error(f"Error: {e}")
            return
        elapsed = time.time() - start_time
        status.success(f"✅ Done in {format_time(elapsed)}")

        # Final metrics
        tm = summary.get('text_matches_found', 0)
        tr = summary.get('text_replacements_done', 0)
        im_p = summary.get('images_processed', 0)
        tot_m = summary.get('total_media', 0)
        im_m = summary.get('image_matches_found', 0)
        im_r = summary.get('image_replacements_done', 0)
        st.info(
            f"Final: Text matches: {tm}, Text appended: {tr}; "
            f"Images processed: {im_p}/{tot_m}, Image matches: {im_m}, Image replacements: {im_r}"
        )

        with st.expander("📝 Modified Text Preview", expanded=False):
            try:
                mod_text = extract_text_from_docx(out_path)
            except Exception as e:
                mod_text = f"Error: {e}"
            st.text_area("Modified text", mod_text, height=200)

        status.text("🔍 Generating diff...")
        try:
            diff_html = highlight_differences(orig_text, mod_text)
        except Exception as e:
            diff_html = f"Error generating diff: {e}"
        with st.expander("🗒️ Inline Diff", expanded=True):
            st.markdown(diff_html, unsafe_allow_html=True)

        st.success(f"Processed file saved at: {os.path.abspath(out_path)}")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed .docx",
                data=f,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

def batch_processing_page(patterns, ocr_langs, include_images, use_multiple_ocr):
    st.header("📄 Batch Processing")
    st.markdown("""
    Upload multiple `.doc` or `.docx` files for processing.
    Each file will be converted (if `.doc`), processed, and saved under the specified output folder on the server as `processed_<original_stem>.docx`.
    """)
    uploaded_files = st.file_uploader(
        "Select multiple documents", 
        type=["doc", "docx"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    if not uploaded_files:
        st.info("Upload one or more files to proceed.")
        return

    output_dir = st.text_input("Output directory on server", value="output_docs/batch_output")
    os.makedirs(output_dir, exist_ok=True)
    st.info(f"Processed files will be saved under: `{os.path.abspath(output_dir)}`")

    if st.button("🔁 Process Documents (Batch)"):
        total = len(uploaded_files)
        global_progress = st.progress(0.0)
        global_status = st.empty()
        start_time = time.time()
        results = []
        for idx, uploaded in enumerate(uploaded_files):
            file_start = time.time()
            global_status.info(f"Processing {idx+1}/{total}: {uploaded.name}")
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpf:
                tmpf.write(uploaded.getbuffer())
                tmp_path = tmpf.name
            converted_path = tmp_path
            if suffix == ".doc":
                try:
                    converted = attempt_doc_to_docx(tmp_path)
                    converted_path = converted
                except Exception as e:
                    results.append({
                        'file': uploaded.name,
                        'status': f'❌ Conversion failed: {e}',
                        'time': '0s',
                        'path': None,
                        'text_matches': 0,
                        'text_appended': 0,
                        'images_processed': 0,
                        'total_media': 0,
                        'image_matches': 0,
                        'image_replacements': 0
                    })
                    try: os.unlink(tmp_path)
                    except: pass
                    global_progress.progress((idx+1)/total)
                    continue
            out_name = f"processed_{Path(uploaded.name).stem}.docx"
            out_path = os.path.join(output_dir, out_name)

            try:
                summary = replace_patterns_in_docx(
                    input_path=converted_path,
                    output_path=out_path,
                    patterns=patterns,
                    ocr_langs=ocr_langs,
                    include_images=include_images,
                    use_multiple_ocr=use_multiple_ocr,
                    progress_callback=None  # skip live callback in batch
                )
                elapsed = time.time() - file_start
                results.append({
                    'file': uploaded.name,
                    'status': '✅ Success',
                    'time': format_time(elapsed),
                    'path': out_path,
                    'text_matches': summary.get('text_matches_found', 0),
                    'text_appended': summary.get('text_replacements_done', 0),
                    'images_processed': summary.get('images_processed', 0),
                    'total_media': summary.get('total_media', 0),
                    'image_matches': summary.get('image_matches_found', 0),
                    'image_replacements': summary.get('image_replacements_done', 0)
                })
            except Exception as e:
                elapsed = time.time() - file_start
                results.append({
                    'file': uploaded.name,
                    'status': f'❌ Error: {e}',
                    'time': format_time(elapsed),
                    'path': None,
                    'text_matches': 0,
                    'text_appended': 0,
                    'images_processed': 0,
                    'total_media': 0,
                    'image_matches': 0,
                    'image_replacements': 0
                })
            finally:
                try: os.unlink(tmp_path)
                except: pass
                if converted_path != tmp_path:
                    try: os.unlink(converted_path)
                    except: pass
            global_progress.progress((idx+1)/total)
        total_elapsed = time.time() - start_time
        st.success(f"Completed {total} files in {format_time(total_elapsed)}")

        # Show results and downloads
        for res in results:
            if res['status'].startswith('✅') and res['path'] and os.path.exists(res['path']):
                st.write(f"✅ {res['file']} (time: {res['time']})")
                st.write(f"   • Text matches: {res['text_matches']}, Text appended: {res['text_appended']}")
                st.write(f"   • Images processed: {res['images_processed']}/{res['total_media']}, "
                         f"Image matches: {res['image_matches']}, Image replacements: {res['image_replacements']}")
                with open(res['path'], "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {Path(res['path']).name}",
                        data=f,
                        file_name=Path(res['path']).name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"dl_{res['file']}"
                    )
            else:
                st.write(f"{res['status']} for {res['file']} (time: {res['time']})")

        # ZIP of all successful processed files
        zip_name = "batch_results.zip"
        zip_path = os.path.join(output_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for res in results:
                if res['status'].startswith('✅') and res['path'] and os.path.exists(res['path']):
                    z.write(res['path'], arcname=Path(res['path']).name)
        st.info(f"All successful processed files zipped as `{zip_name}` in `{os.path.abspath(output_dir)}`")
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Batch ZIP",
                data=f,
                file_name=zip_name,
                mime="application/zip"
            )

def comparison_page():
    st.header("🔍 Document Comparison")
    st.markdown("""
    Upload sets of original and processed DOCX files.  
    Processed files should be named `processed_<original_stem>.docx` or share the same stem.
    The app will match by stem and show inline text diffs and side-by-side image comparisons.
    """)
    col1, col2 = st.columns(2)
    with col1:
        original_files = st.file_uploader(
            "Upload original DOCX files",
            type=["docx"],
            accept_multiple_files=True,
            key="cmp_orig"
        )
        if original_files:
            st.success(f"Selected {len(original_files)} original files")
    with col2:
        processed_files = st.file_uploader(
            "Upload processed DOCX files",
            type=["docx"],
            accept_multiple_files=True,
            key="cmp_proc"
        )
        if processed_files:
            st.success(f"Selected {len(processed_files)} processed files")
    if original_files and processed_files:
        if st.button("🔍 Compare Documents"):
            matched_pairs, unmatched_orig, unmatched_proc = match_files(original_files, processed_files)
            st.subheader("File Matching Results")
            colm, colu1, colu2 = st.columns(3)
            with colm:
                st.metric("Matched Pairs", len(matched_pairs))
            with colu1:
                st.metric("Unmatched Originals", len(unmatched_orig))
                if unmatched_orig:
                    with st.expander("View unmatched originals"):
                        for f in unmatched_orig:
                            st.write(f.name)
            with colu2:
                st.metric("Unmatched Processed", len(unmatched_proc))
                if unmatched_proc:
                    with st.expander("View unmatched processed"):
                        for f in unmatched_proc:
                            st.write(f.name)
            if matched_pairs:
                st.subheader("Comparisons")
                progress = st.progress(0)
                status = st.empty()
                results = []
                total = len(matched_pairs)
                for idx, (of, pf) in enumerate(matched_pairs):
                    status.info(f"Comparing {idx+1}/{total}: {of.name}")
                    res = compare_document_pair(of, pf)
                    results.append(res)
                    progress.progress((idx+1)/total)
                status.success("Completed comparisons")
                for res in results:
                    with st.expander(f"{res['original']} ↔ {res['processed']}", expanded=False):
                        if res['error']:
                            st.error(f"Error: {res['error']}")
                            continue
                        st.subheader("Text Differences")
                        st.markdown(
                            f"<div style='border:1px solid #e0e0e0; padding:10px; border-radius:5px; max-height:300px; overflow-y:auto'>{res['text_diff_html']}</div>",
                            unsafe_allow_html=True
                        )
                        if res['orig_images'] or res['proc_images']:
                            st.subheader("Image Comparison")
                            max_imgs = max(len(res['orig_images']), len(res['proc_images']))
                            for i in range(max_imgs):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.text(f"Original Image {i+1}")
                                    if i < len(res['orig_images']):
                                        img = Image.open(io.BytesIO(res['orig_images'][i]))
                                        st.image(img, use_container_width=True)
                                    else:
                                        st.write("No image")
                                with c2:
                                    st.text(f"Processed Image {i+1}")
                                    if i < len(res['proc_images']):
                                        img = Image.open(io.BytesIO(res['proc_images'][i]))
                                        st.image(img, use_container_width=True)
                                    else:
                                        st.write("No image")
                                st.markdown("---")
            else:
                st.warning("No matching pairs found")

# ---------------- Main UI ----------------

def main():
    st.sidebar.title("Settings")

    # Windows-specific Tesseract auto-detect and manual override
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(Path.home() / r"AppData\Local\Tesseract-OCR\tesseract.exe"),
            r"C:\tesseract\tesseract.exe"
        ]
        found = False
        for p in possible_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                found = True
                break
        if found:
            st.sidebar.success(f"Detected Tesseract at: {pytesseract.pytesseract.tesseract_cmd}")
        else:
            st.sidebar.warning("Tesseract not found in common Windows paths.")
        custom = st.sidebar.text_input("Tesseract executable path (Windows)", value="", help="If auto-detect fails, specify full path to tesseract.exe")
        if custom:
            if os.path.exists(custom):
                pytesseract.pytesseract.tesseract_cmd = custom
                st.sidebar.success(f"Tesseract path set to: {custom}")
            else:
                st.sidebar.error("Provided Tesseract path not found")

    # Check OCR engine availability
    ocr_status = {
        "Tesseract": False,
        "PaddleOCR": PADDLEOCR_AVAILABLE,
        "EasyOCR": EASYOCR_AVAILABLE
    }
    
    try:
        tess_ver = pytesseract.get_tesseract_version()
        st.sidebar.success(f"Tesseract v{tess_ver} detected")
        ocr_status["Tesseract"] = True
    except Exception:
        st.sidebar.error("Tesseract not found or not configured. Image OCR may be limited.")
    
    if PADDLEOCR_AVAILABLE:
        st.sidebar.success("PaddleOCR is available")
    else:
        st.sidebar.warning("PaddleOCR not installed. Run 'pip install paddleocr paddlepaddle' to enable.")
    
    if EASYOCR_AVAILABLE:
        st.sidebar.success("EasyOCR is available")
    else:
        st.sidebar.warning("EasyOCR not installed. Run 'pip install easyocr' to enable.")
    
    if not any(ocr_status.values()):
        st.sidebar.error("No OCR engines available. Image processing will be disabled.")

    # Navigation
    st.sidebar.divider()
    st.sidebar.title("Mode")
    mode = st.sidebar.radio("Select Mode", ["Single File", "Batch Processing", "Document Comparison"])

    # Load patterns once
    patterns = load_patterns()

    # OCR settings
    st.sidebar.subheader("OCR Settings")
    ocr_langs = st.sidebar.text_input("Tesseract Languages", value="eng+chi_sim+chi_tra", 
                                     help="Language codes separated by '+' (e.g., eng+chi_sim+chi_tra)")
    
    # Image processing options
    include_images = st.sidebar.checkbox("Process Images", value=True, 
                                       help="Enable OCR-based pattern detection in images")
    
    use_multiple_ocr = st.sidebar.checkbox("Use Multiple OCR Engines", value=True,
                                          help="Combine Tesseract, PaddleOCR and EasyOCR for better detection")
    
    if not any(ocr_status.values()):
        include_images = False
        use_multiple_ocr = False

    if mode == "Single File":
        single_file_page(patterns, ocr_langs, include_images, use_multiple_ocr)
    elif mode == "Batch Processing":
        batch_processing_page(patterns, ocr_langs, include_images, use_multiple_ocr)
    elif mode == "Document Comparison":
        comparison_page()

if __name__ == "__main__":
    main()