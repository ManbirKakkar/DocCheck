import streamlit as st
import os
import re
import tempfile
import zipfile
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytesseract
import time
from datetime import timedelta
import base64
from PIL import Image
from io import BytesIO
import difflib

# Set default output path
DEFAULT_OUTPUT_PATH = "output_docs"

# Configure Tesseract path if needed (uncomment and set path for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="DOCX Number Processor", layout="wide")

def create_output_dir(path):
    """Create output directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def process_image(image_bytes):
    """Process image to replace number patterns using OCR"""
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB and get dimensions
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape
        
        # Use Tesseract to detect text and bounding boxes
        data = pytesseract.image_to_data(
            img_rgb, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )
        
        # Define pattern and replacement
        pattern = re.compile(r'\b72-(\d{3,4}-\d{7}-\d{2})\b')
        replacement = r'4022-\1'
        
        # Process each detected text element
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Confidence threshold
                text = data['text'][i]
                if pattern.search(text):
                    # Get coordinates
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Create background patch to cover original text
                    roi = img[y-2:y+h+2, x-2:x+w+2]  # Slightly larger region
                    if roi.size > 0:
                        # Use average color for background
                        avg_color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                        cv2.rectangle(img, (x-2, y-2), (x+w+2, y+h+2), avg_color.tolist(), -1)
                        
                        # Prepare new text
                        new_text = pattern.sub(replacement, text)
                        
                        # Calculate font size based on bounding box height
                        font_scale = h / 35
                        thickness = max(1, int(font_scale))
                        
                        # Draw new text
                        cv2.putText(
                            img, 
                            new_text, 
                            (x, y+h),  # Position at bottom of original text area
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            (0, 0, 0),  # Black text
                            thickness,
                            cv2.LINE_AA
                        )
        
        # Convert back to bytes
        _, img_bytes = cv2.imencode('.jpg', img)
        return img_bytes.tobytes()
    
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return image_bytes  # Return original if processing fails

def replace_pattern_in_docx(docx_path, output_path, progress_callback=None):
    """Process DOCX file to handle number patterns in text and images"""
    pattern = re.compile(r'\b72-(\d{3,4}-\d{7}-\d{2})\b')
    
    # Create temporary working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract DOCX contents
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Process text in document, headers, and footers
        xml_files = [
            'word/document.xml',
            *[f'word/header{i}.xml' for i in range(1, 4)],
            *[f'word/footer{i}.xml' for i in range(1, 4)]
        ]
        
        # Text processing progress
        total_text_files = len(xml_files)
        processed_files = 0
        
        for xml_file in xml_files:
            file_path = tmpdir / xml_file
            if not file_path.exists():
                continue
                
            # Parse XML and process text
            tree = ET.parse(file_path)
            root = tree.getroot()
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Find all text elements
            for t in root.findall('.//w:t', namespaces):
                if t.text and pattern.search(t.text):
                    # For text: append new number after comma
                    t.text = pattern.sub(r'\g<0>,4022-\1', t.text)
            
            # Save changes
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            # Update progress
            processed_files += 1
            if progress_callback:
                progress_callback(50 * processed_files / total_text_files)
        
        # Process images in media directory
        media_dir = tmpdir / 'word' / 'media'
        if media_dir.exists():
            image_files = list(media_dir.glob('*.*'))
            total_images = len(image_files)
            
            for idx, img_file in enumerate(image_files):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    with open(img_file, 'rb') as f:
                        img_bytes = f.read()
                    
                    # Process image
                    processed_img = process_image(img_bytes)
                    
                    # Save processed image
                    with open(img_file, 'wb') as f:
                        f.write(processed_img)
                
                # Update progress
                if progress_callback:
                    progress_callback(50 + 50 * (idx + 1) / total_images)
        
        # Create new DOCX with modified content
        with zipfile.ZipFile(output_path, 'w') as zip_out:
            for root_path, _, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root_path, file)
                    arcname = os.path.relpath(file_path, tmpdir)
                    zip_out.write(file_path, arcname)

def extract_text_from_docx(docx_path):
    """Extract text content from DOCX file"""
    text = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract DOCX contents
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Process text in document, headers, and footers
        xml_files = [
            'word/document.xml',
            *[f'word/header{i}.xml' for i in range(1, 4)],
            *[f'word/footer{i}.xml' for i in range(1, 4)]
        ]
        
        for xml_file in xml_files:
            file_path = tmpdir / xml_file
            if not file_path.exists():
                continue
                
            # Parse XML and extract text
            tree = ET.parse(file_path)
            root = tree.getroot()
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Find all text elements
            for t in root.findall('.//w:t', namespaces):
                if t.text:
                    text += t.text + " "
                    
    return text

def extract_images_from_docx(docx_path):
    """Extract images from DOCX file"""
    images = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract DOCX contents
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Process images in media directory
        media_dir = tmpdir / 'word' / 'media'
        if media_dir.exists():
            image_files = list(media_dir.glob('*.*'))
            
            for img_file in image_files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    with open(img_file, 'rb') as f:
                        images.append(f.read())
                    
    return images

def highlight_differences(text1, text2):
    """Create a visual diff of two text documents"""
    d = difflib.Differ()
    diff = list(d.compare(text1.split(), text2.split()))
    
    result = []
    for word in diff:
        if word.startswith('  '):
            # Unchanged
            result.append(word[2:])
        elif word.startswith('- '):
            # Removed
            result.append(f"<span style='background-color: #ffcccc; text-decoration: line-through;'>{word[2:]}</span>")
        elif word.startswith('+ '):
            # Added
            result.append(f"<span style='background-color: #ccffcc;'>{word[2:]}</span>")
    
    return " ".join(result)

def batch_processing_page():
    st.title("📄 Batch Processing")
    st.markdown("""
    ### Process multiple DOCX files with pattern replacement:
    - Upload multiple files at once
    - Process all files in a single operation
    - Each file processed independently
    """)
    
    # Dependencies note
    with st.expander("⚠️ Important Requirements", expanded=True):
        st.markdown("""
        **For image text replacement:**
        1. Install Tesseract OCR:
           - **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
           - **Mac**: `brew install tesseract`
           - **Linux**: `sudo apt install tesseract-ocr`
        2. Install Python dependencies:
           ```bash
           pip install opencv-python pytesseract numpy
           ```
        """)
    
    # Create default output directory
    output_path = create_output_dir(DEFAULT_OUTPUT_PATH)
    
    # File upload section
    with st.expander("📤 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select DOCX files", 
            type=['docx'],
            help="Documents containing 72-xxx-xxxxxxx-xx patterns",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files for processing")
    
    # Output configuration
    with st.expander("⚙️ Output Settings", expanded=True):
        output_dir = st.text_input(
            "Output Directory", 
            value=DEFAULT_OUTPUT_PATH,
            help="Folder where processed files will be saved"
        )
        output_dir = create_output_dir(output_dir)
        st.info(f"Processed files will be saved in: `{os.path.abspath(output_dir)}`")
    
    # Processing section
    if uploaded_files:
        if st.button("🔁 Process Documents", use_container_width=True, type="primary"):
            # Setup progress indicators
            global_progress_bar = st.progress(0)
            global_status = st.empty()
            global_time_area = st.empty()
            results_container = st.container()
            
            # Start timer
            start_time = time.time()
            
            # Create results table
            results = []
            
            # Process each file
            total_files = len(uploaded_files)
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                file_start_time = time.time()
                
                # Update global status
                global_status.info(f"📁 Processing file {file_idx+1}/{total_files}: {uploaded_file.name}")
                global_progress_bar.progress((file_idx) / total_files)
                
                # Prepare output filename
                original_name = uploaded_file.name
                processed_name = f"processed_{original_name}"
                output_filepath = os.path.join(output_dir, processed_name)
                
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Create progress callback for individual file
                file_progress_bar = results_container.progress(0)
                file_status = results_container.empty()
                file_time = results_container.empty()
                
                def update_file_progress(percent, file_idx=file_idx):
                    file_progress_bar.progress(min(100, int(percent)))
                    elapsed = time.time() - file_start_time
                    file_time.info(f"⏱️ File time: {timedelta(seconds=int(elapsed))}")
                
                # Process document
                try:
                    file_status.info(f"🔄 Starting processing: {uploaded_file.name}")
                    replace_pattern_in_docx(tmp_path, output_filepath, progress_callback=update_file_progress)
                    
                    # Record success
                    file_elapsed = time.time() - file_start_time
                    file_status.success(f"✅ Processed in {timedelta(seconds=file_elapsed)}")
                    file_progress_bar.progress(100)
                    
                    # Add to results
                    results.append({
                        'file': original_name,
                        'status': '✅ Success',
                        'time': str(timedelta(seconds=int(file_elapsed))),
                        'path': output_filepath,
                        'processed_name': processed_name
                    })
                    
                except Exception as e:
                    file_elapsed = time.time() - file_start_time
                    file_status.error(f"❌ Error processing: {str(e)}")
                    
                    # Add to results
                    results.append({
                        'file': original_name,
                        'status': f'❌ Error: {str(e)[:50]}...',
                        'time': str(timedelta(seconds=int(file_elapsed))),
                        'path': '',
                        'processed_name': ''
                    })
                
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                # Update global progress
                global_progress_bar.progress((file_idx+1) / total_files)
                elapsed = time.time() - start_time
                global_time_area.info(f"⏱️ Total elapsed time: {timedelta(seconds=int(elapsed))}")
            
            # Final update
            elapsed = time.time() - start_time
            global_time_area.success(f"✅ All files processed in {timedelta(seconds=elapsed)}")
            global_status.success(f"✅ Processed {total_files} files")
            
            # Show results table
            with results_container.expander("📊 Processing Results", expanded=True):
                if results:
                    # Create download buttons for successful files
                    for result in results:
                        if result['status'] == '✅ Success':
                            with open(result['path'], 'rb') as f:
                                st.download_button(
                                    label=f"⬇️ Download {result['processed_name']}",
                                    data=f,
                                    file_name=result['processed_name'],
                                    mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                    key=f"download_{result['file']}"
                                )
                    
                    # Show summary table
                    st.subheader("Processing Summary")
                    st.table([{
                        "File": r['file'],
                        "Status": r['status'],
                        "Processing Time": r['time'],
                    } for r in results])

def match_files(original_files, processed_files):
    """Match original files with processed files based on filename pattern"""
    matched_pairs = []
    unmatched_originals = []
    unmatched_processed = []
    
    # Create a mapping of processed filenames without the prefix
    processed_map = {}
    for file in processed_files:
        name = file.name
        if name.startswith("processed_"):
            base_name = name[10:]
            processed_map[base_name] = file
        else:
            unmatched_processed.append(file)
    
    # Match original files with processed files
    for file in original_files:
        if file.name in processed_map:
            matched_pairs.append((file, processed_map[file.name]))
        else:
            unmatched_originals.append(file)
    
    return matched_pairs, unmatched_originals, unmatched_processed

def compare_document_pair(original_file, processed_file):
    """Compare a single document pair and return results"""
    with st.spinner(f"Comparing {original_file.name} and {processed_file.name}..."):
        # Save files to temp locations
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as orig_tmp:
            orig_tmp.write(original_file.getbuffer())
            orig_path = orig_tmp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as proc_tmp:
            proc_tmp.write(processed_file.getbuffer())
            proc_path = proc_tmp.name
        
        try:
            # Extract text from both documents
            orig_text = extract_text_from_docx(orig_path)
            proc_text = extract_text_from_docx(proc_path)
            
            # Extract images from both documents
            orig_images = extract_images_from_docx(orig_path)
            proc_images = extract_images_from_docx(proc_path)
            
            # Create visual diff
            html_diff = highlight_differences(orig_text, proc_text)
            
            return {
                'original': original_file.name,
                'processed': processed_file.name,
                'text_diff': html_diff,
                'orig_images': orig_images,
                'proc_images': proc_images,
                'error': None
            }
            
        except Exception as e:
            return {
                'original': original_file.name,
                'processed': processed_file.name,
                'text_diff': "",
                'orig_images': [],
                'proc_images': [],
                'error': str(e)
            }
            
        finally:
            # Clean up temp files
            if os.path.exists(orig_path):
                os.unlink(orig_path)
            if os.path.exists(proc_path):
                os.unlink(proc_path)

def comparison_page():
    st.title("🔍 Batch Document Comparison")
    st.markdown("""
    ### Compare multiple original and processed DOCX files:
    - Upload sets of original and processed files
    - Files are automatically matched by filename
    - See differences for each document pair
    """)
    
    st.info("Processed files should have the naming pattern: `processed_<original_filename>.docx`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Documents")
        original_files = st.file_uploader(
            "Upload original DOCX files", 
            type=['docx'],
            key="original_uploader",
            accept_multiple_files=True
        )
        
        if original_files:
            st.success(f"Selected {len(original_files)} original files")
    
    with col2:
        st.subheader("Processed Documents")
        processed_files = st.file_uploader(
            "Upload processed DOCX files", 
            type=['docx'],
            key="processed_uploader",
            accept_multiple_files=True
        )
        
        if processed_files:
            st.success(f"Selected {len(processed_files)} processed files")
    
    if original_files and processed_files:
        if st.button("🔍 Compare Documents", use_container_width=True, type="primary"):
            # Match files based on name
            matched_pairs, unmatched_originals, unmatched_processed = match_files(
                original_files, processed_files
            )
            
            # Show matching results
            st.subheader("File Matching Results")
            col_match, col_orig, col_proc = st.columns(3)
            
            with col_match:
                st.metric("Matched Pairs", len(matched_pairs))
            
            with col_orig:
                st.metric("Unmatched Originals", len(unmatched_originals))
                if unmatched_originals:
                    with st.expander("View unmatched originals"):
                        for file in unmatched_originals:
                            st.write(file.name)
            
            with col_proc:
                st.metric("Unmatched Processed", len(unmatched_processed))
                if unmatched_processed:
                    with st.expander("View unmatched processed"):
                        for file in unmatched_processed:
                            st.write(file.name)
            
            # Process all matched pairs
            if matched_pairs:
                st.subheader("Document Comparisons")
                st.info(f"Found {len(matched_pairs)} matching document pairs")
                
                # Setup progress
                total_pairs = len(matched_pairs)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create container for results
                results_container = st.container()
                
                # Process each pair
                comparison_results = []
                for idx, (orig_file, proc_file) in enumerate(matched_pairs):
                    status_text.info(f"Comparing pair {idx+1}/{total_pairs}: {orig_file.name}")
                    progress_bar.progress((idx) / total_pairs)
                    
                    # Compare the document pair
                    result = compare_document_pair(orig_file, proc_file)
                    comparison_results.append(result)
                    
                    # Update progress
                    progress_bar.progress((idx+1) / total_pairs)
                
                # Display results
                status_text.success(f"✅ Completed {total_pairs} comparisons")
                
                # Show each comparison in an expander
                for result in comparison_results:
                    with results_container.expander(f"🔍 {result['original']} ↔ {result['processed']}", expanded=False):
                        if result['error']:
                            st.error(f"Comparison error: {result['error']}")
                            continue
                        
                        # Show text comparison
                        st.subheader("Text Differences")
                        st.markdown(f"<div style='border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{result['text_diff']}</div>", 
                                    unsafe_allow_html=True)
                        
                        # Show image comparison
                        if result['orig_images'] or result['proc_images']:
                            st.subheader("Image Comparison")
                            
                            # Determine max number of images to show
                            max_images = max(len(result['orig_images']), len(result['proc_images']))
                            
                            # Create columns for each image pair
                            for i in range(max_images):
                                col_orig, col_proc = st.columns(2)
                                
                                with col_orig:
                                    st.subheader(f"Original Image {i+1}")
                                    if i < len(result['orig_images']):
                                        st.image(result['orig_images'][i], use_column_width=True)
                                    else:
                                        st.warning("No image")
                                
                                with col_proc:
                                    st.subheader(f"Processed Image {i+1}")
                                    if i < len(result['proc_images']):
                                        st.image(result['proc_images'][i], use_column_width=True)
                                    else:
                                        st.warning("No image")
                                
                                st.divider()
            else:
                st.warning("No matching document pairs found")

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", 
                                ["Batch Processing", "Document Comparison"],
                                index=0)
    
    # Dependencies note in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Requirements:**
    - Tesseract OCR installed
    - Python packages:
      ```bash
      pip install opencv-python pytesseract numpy
      ```
    """)
    
    # Run the selected page
    if app_mode == "Batch Processing":
        batch_processing_page()
    elif app_mode == "Document Comparison":
        comparison_page()
    
    # Show sample pattern
    st.sidebar.divider()

if __name__ == "__main__":
    main()