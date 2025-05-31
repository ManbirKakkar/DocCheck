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
import platform
import html
from pdf2docx import Converter
import fitz  # For PDF text extraction

# Set default output path
DEFAULT_OUTPUT_PATH = "output_docs"

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    # Check common installation paths
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getlogin()),
        r"C:\tesseract\tesseract.exe"
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            break
    
    if not tesseract_found:
        st.warning("Tesseract not found in common Windows paths. OCR functionality may not work.")
else:
    # Assume Tesseract is in PATH for Linux/Mac
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        st.warning("Tesseract not found in system PATH. OCR functionality may not work.")

st.set_page_config(page_title="Document Number Processor", layout="wide")

# Unified pattern for all formats - FIXED
COMBINED_PATTERN = re.compile(
    r'\b77\s*[-–—]?\s*([a-zA-Z0-9]{2,3})\s*[-–—]?\s*([a-zA-Z0-9]{5,7}[-a-zA-Z0-9]*)(?:\s*[-–—]?\s*([a-zA-Z0-9]{2,4}))?\b',
    re.IGNORECASE
)

def create_output_dir(path):
    """Create output directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def convert_pdf_to_docx(pdf_path, docx_path):
    """Convert PDF to DOCX for processing"""
    try:
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        return True
    except Exception as e:
        st.error(f"PDF conversion failed: {str(e)}")
        return False

def process_image(image_bytes):
    """Process image to replace number patterns using OCR"""
    try:
        # Try to get Tesseract version to verify installation
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract OCR is not installed or not in your PATH. Image processing disabled.")
        return image_bytes
        
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return image_bytes
            
        # Preprocessing for better OCR
        # Resize image for better recognition
        scale_factor = 1.5
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Use Tesseract to detect text and bounding boxes - FIXED CONFIG
        data = pytesseract.image_to_data(
            gray, 
            output_type=pytesseract.Output.DICT,
            config='--psm 4 --oem 3 -l eng+chi_sim'  # Multi-language support
        )
        
        # Track processed regions to avoid overlapping
        processed_regions = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Confidence threshold
                text = data['text'][i]
                match = COMBINED_PATTERN.search(text)
                
                if match:
                    # Get coordinates
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Skip if overlapping with previous processing
                    overlap = False
                    for (px, py, pw, ph) in processed_regions:
                        if (x < px + pw and x + w > px and
                            y < py + ph and y + h > py):
                            overlap = True
                            break
                    
                    if overlap:
                        continue
                    
                    # Create background patch to cover original text
                    padding = 5
                    roi = img[y-padding:y+h+padding, x-padding:x+w+padding]
                    if roi.size > 0:
                        # Use average color for background
                        avg_color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                        cv2.rectangle(img, (x-padding, y-padding), (x+w+padding, y+h+padding), avg_color.tolist(), -1)
                        
                        # Prepare new text - HANDLE COMPLEX CASES
                        g1, g2, g3 = match.group(1), match.group(2), match.group(3)
                        
                        # Handle cases like 77-531-014BLK-245
                        if '-' in g2 and not g3:
                            parts = g2.split('-')
                            new_text = f"4022-{g1}-{parts[0]}"
                            if len(parts) > 1:
                                new_text += f"-{''.join(parts[1:])}"
                        else:
                            new_text = f"4022-{g1}-{g2}"
                            if g3:
                                new_text += f"-{g3}"
                            
                        # Calculate font size based on bounding box height
                        font_scale = h / 35
                        thickness = max(1, int(font_scale))
                        
                        # Calculate text size to center it
                        text_size = cv2.getTextSize(
                            new_text, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            thickness
                        )[0]
                        
                        text_x = x + (w - text_size[0]) // 2
                        text_y = y + h // 2 + text_size[1] // 2
                        
                        # Draw new text
                        cv2.putText(
                            img, 
                            new_text, 
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            (0, 0, 0),  # Black text
                            thickness,
                            cv2.LINE_AA
                        )
                        
                        # Track processed region
                        processed_regions.append((x-padding, y-padding, w+padding*2, h+padding*2))
        
        # Convert back to bytes
        _, img_bytes = cv2.imencode('.jpg', img)
        return img_bytes.tobytes()
    
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return image_bytes  # Return original if processing fails

def replace_pattern_in_docx(docx_path, output_path, progress_callback=None):
    """Process DOCX file to handle number patterns in text and images"""
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
                if t.text:
                    # Replace patterns only once - FIXED LOGIC
                    def replace_func(match):
                        a, b, c = match.group(1), match.group(2), match.group(3)
                        base = f"77-{a}-{b}"
                        new = f"4022-{a}-{b.split('-')[0]}"  # Take only the first part before hyphen
                        
                        # Handle cases like 77-531-014BLK-245
                        if '-' in b and not c:
                            parts = b.split('-')
                            base = f"77-{a}-{'-'.join(parts)}"
                            new = f"4022-{a}-{parts[0]}"
                            if len(parts) > 1:
                                new += f"-{''.join(parts[1:])}"
                        
                        if c:
                            base += f"-{c}"
                            new += f"-{c}"
                        return f"{base},{new}"
                    
                    t.text = COMBINED_PATTERN.sub(replace_func, t.text)
            
            # Enable auto-size for textboxes to prevent clipping
            for textbox in root.findall('.//w:txbxContent', namespaces):
                bodyPr = textbox.find('.//w:bodyPr', namespaces)
                if bodyPr is not None:
                    # Enable auto-sizing
                    bodyPr.set('vertOverflow', "overflow")
                    bodyPr.set('horzOverflow', "overflow")
                    bodyPr.set('wrap', "none")
            
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

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"PDF text extraction failed: {str(e)}")
        return ""

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
    import difflib
    d = difflib.Differ()
    diff = list(d.compare(text1.split(), text2.split()))
    
    result = []
    for word in diff:
        if word.startswith('  '):
            # Unchanged - escape HTML
            result.append(html.escape(word[2:]))
        elif word.startswith('- '):
            # Removed - escape HTML
            escaped_word = html.escape(word[2:])
            result.append(f"<span style='background-color: #ffcccc; text-decoration: line-through;'>{escaped_word}</span>")
        elif word.startswith('+ '):
            # Added - escape HTML
            escaped_word = html.escape(word[2:])
            result.append(f"<span style='background-color: #ccffcc;'>{escaped_word}</span>")
    
    return " ".join(result)

def batch_processing_page():
    st.title("📄 Batch Processing")
    st.markdown("""
    ### Process multiple DOCX/PDF files with pattern replacement:
    - Upload multiple files at once
    - Process all files in a single operation
    - Each file processed independently
    - Supports various pattern formats
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
           pip install opencv-python pytesseract numpy pdf2docx pymupdf
           ```
        """)
        
        # Tesseract path configuration for Windows
        if platform.system() == "Windows":
            st.markdown("### Windows Tesseract Configuration")
            tesseract_path = pytesseract.pytesseract.tesseract_cmd if hasattr(pytesseract.pytesseract, 'tesseract_cmd') else "Not configured"
            st.info(f"Tesseract is installed in: {tesseract_path}")
            
            if st.button("Refresh Tesseract Detection"):
                st.experimental_rerun()
    
    # Create default output directory
    output_path = create_output_dir(DEFAULT_OUTPUT_PATH)
    
    # File upload section
    with st.expander("📤 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select DOCX or PDF files", 
            type=['docx', 'pdf'],
            help="Documents containing patterns like 77-xxx-xxxxxx",
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
                docx_temp = None  # Initialize docx_temp here
                
                # Update global status
                global_status.info(f"📁 Processing file {file_idx+1}/{total_files}: {uploaded_file.name}")
                global_progress_bar.progress((file_idx) / total_files)
                
                # Prepare output filename
                original_name = uploaded_file.name
                base_name = Path(original_name).stem
                output_filepath = os.path.join(output_dir, f"processed_{base_name}.docx")
                
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{Path(original_name).suffix}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Handle PDF files
                if uploaded_file.name.lower().endswith('.pdf'):
                    # Convert PDF to DOCX
                    docx_temp = tmp_path.replace('.pdf', '.docx')
                    if convert_pdf_to_docx(tmp_path, docx_temp):
                        processing_path = docx_temp
                    else:
                        # Skip processing if conversion failed
                        results.append({
                            'file': original_name,
                            'status': '❌ PDF conversion failed',
                            'time': '0:00:00',
                            'path': '',
                            'processed_name': ''
                        })
                        continue
                else:
                    processing_path = tmp_path
                
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
                    replace_pattern_in_docx(processing_path, output_filepath, progress_callback=update_file_progress)
                    
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
                        'processed_name': f"processed_{base_name}.docx"
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
                    # Clean up temp files
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    if docx_temp and os.path.exists(docx_temp):
                        os.unlink(docx_temp)
                
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
                        if result['status'] == '✅ Success' and os.path.exists(result['path']):
                            with open(result['path'], 'rb') as f:
                                st.download_button(
                                    label=f"⬇️ Download {result['processed_name']}",
                                    data=f,
                                    file_name=result['processed_name'],
                                    mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                    key=f"download_{result['file']}_{time.time()}"
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
    
    # Create a mapping of processed filenames without the prefix and extension
    processed_map = {}
    for file in processed_files:
        name = file.name
        if name.startswith("processed_"):
            # Remove the prefix and get the stem (without extension)
            base_name = Path(name[10:]).stem
            processed_map[base_name] = file
        else:
            unmatched_processed.append(file)
    
    # Match original files with processed files
    for file in original_files:
        # Get stem without extension
        base_name = Path(file.name).stem
        
        if base_name in processed_map:
            matched_pairs.append((file, processed_map[base_name]))
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
            if original_file.name.lower().endswith('.pdf'):
                orig_text = extract_text_from_pdf(orig_path)
            else:
                orig_text = extract_text_from_docx(orig_path)
                
            if processed_file.name.lower().endswith('.pdf'):
                proc_text = extract_text_from_pdf(proc_path)
            else:
                proc_text = extract_text_from_docx(proc_path)
            
            # Extract images from both documents
            orig_images = extract_images_from_docx(orig_path) if not original_file.name.lower().endswith('.pdf') else []
            proc_images = extract_images_from_docx(proc_path) if not processed_file.name.lower().endswith('.pdf') else []
            
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
    ### Compare multiple original and processed DOCX/PDF files:
    - Upload sets of original and processed files
    - Files are automatically matched by filename
    - See differences for each document pair
    """)
    
    st.info("Processed files should have the naming pattern: `processed_<original_filename>.docx`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Documents")
        original_files = st.file_uploader(
            "Upload original DOCX/PDF files", 
            type=['docx', 'pdf'],
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
                                        st.image(result['orig_images'][i], use_container_width=True)
                                    else:
                                        st.warning("No image")
                                
                                with col_proc:
                                    st.subheader(f"Processed Image {i+1}")
                                    if i < len(result['proc_images']):
                                        st.image(result['proc_images'][i], use_container_width=True)
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
      pip install opencv-python pytesseract numpy pdf2docx pymupdf
      ```
    """)
    
    # Tesseract status
    try:
        version = pytesseract.get_tesseract_version()
        st.sidebar.success(f"Tesseract v{version} detected")
    except pytesseract.TesseractNotFoundError:
        st.sidebar.error("Tesseract not found. Image processing disabled")
    except Exception as e:
        st.sidebar.warning(f"Tesseract detection failed: {str(e)}")
    
    # Pattern examples
    st.sidebar.divider()
    st.sidebar.markdown("### Pattern Examples")
    st.sidebar.markdown("**Supported Formats:**")
    st.sidebar.code("77-ABC-123456\n77-ABC-123456-XX\n77-ABC-123456-XXX\n77-ABC-123456-XXXX")
    st.sidebar.markdown("**Text Processing:**")
    st.sidebar.code("77-ABC-123456 → 77-ABC-123456,4022-ABC-123456")
    st.sidebar.markdown("**Image Processing:**")
    st.sidebar.code("77-ABC-123456 → 4022-ABC-123456")
    
    # Run the selected page
    if app_mode == "Batch Processing":
        batch_processing_page()
    elif app_mode == "Document Comparison":
        comparison_page()

if __name__ == "__main__":
    main()