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

# Page config
st.set_page_config(page_title="Document Number Processor", layout="wide")

# ---------------- Utility functions ----------------

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

def process_image_bytes(image_bytes: bytes, languages: str, compiled_patterns: list) -> bytes:
    """
    OCR on image bytes, apply regex replacements from compiled_patterns (list of (regex, repl)),
    where repl may be a string or callable(match)->str. Return possibly modified image bytes.
    """
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold for better OCR
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        # OCR
        config = f"--psm 4 --oem 3 -l {languages}" if languages else "--psm 4 --oem 3"
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=config)
        n = len(data.get('text', []))
        for i in range(n):
            txt = data['text'][i]
            try:
                conf = int(data['conf'][i])
            except:
                conf = 0
            if conf > 60 and txt.strip():
                x, y, w, h = (data[k][i] for k in ('left', 'top', 'width', 'height'))
                new_txt = txt
                for regex, repl in compiled_patterns:
                    if callable(repl):
                        # replacement function
                        def _fn(m):
                            try:
                                return repl(m)
                            except Exception:
                                return m.group(0)
                        try:
                            new_txt = regex.sub(_fn, new_txt)
                        except re.error:
                            pass
                    else:
                        try:
                            new_txt = regex.sub(repl, new_txt)
                        except re.error:
                            pass
                if new_txt != txt:
                    # Cover original area with white rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    # Write new text
                    font_scale = max(0.5, h / 40)
                    thickness = max(1, int(font_scale))
                    cv2.putText(
                        img,
                        new_txt,
                        (x, y + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )
        success, out = cv2.imencode('.png', img)
        if success:
            return out.tobytes()
        else:
            return image_bytes
    except Exception:
        return image_bytes

def replace_patterns_in_docx(
    input_path: str,
    output_path: str,
    patterns: list,
    ocr_langs: str,
    include_images: bool,
    progress_callback=None
):
    """
    Unpack DOCX at input_path, apply text replacements and optional image OCR replacements,
    then repack to output_path.
    patterns: list of dicts {"pattern": ..., "replacement": ...}, replacement may be string or callable.
    ocr_langs: string for pytesseract languages, e.g. "eng+chi_sim+chi_tra"
    include_images: bool (we always True by default)
    progress_callback: function(percent: float)
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
            # skip invalid
            continue

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Unzip DOCX
        with zipfile.ZipFile(input_path, 'r') as zin:
            zin.extractall(tmpdir)

        # Prepare xml parts
        xml_rel_paths = ['word/document.xml'] + [f"word/header{i}.xml" for i in range(1, 4)] + [f"word/footer{i}.xml" for i in range(1, 4)]
        total_xml = sum(1 for rel in xml_rel_paths if (tmpdir_path / rel).exists())
        media_dir = tmpdir_path / 'word' / 'media'
        total_media = 0
        if include_images and media_dir.exists():
            total_media = sum(1 for f in media_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp'])
        total = total_xml + total_media if (total_xml + total_media) > 0 else 1
        count = 0

        # Text replacements
        for rel in xml_rel_paths:
            xml_file = tmpdir_path / rel
            if not xml_file.exists():
                continue
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                for t in root.findall('.//w:t', ns):
                    if t.text:
                        new_text = t.text
                        for regex, repl in compiled:
                            if callable(repl):
                                def _fn(m):
                                    try:
                                        return repl(m)
                                    except Exception:
                                        return m.group(0)
                                try:
                                    new_text = regex.sub(_fn, new_text)
                                except re.error:
                                    pass
                            else:
                                try:
                                    new_text = regex.sub(repl, new_text)
                                except re.error:
                                    pass
                        t.text = new_text
                tree.write(str(xml_file), encoding='utf-8', xml_declaration=True)
            except Exception:
                pass
            count += 1
            if progress_callback:
                percent = 100 * count / total
                progress_callback(percent)

        # Image OCR replacements
        if include_images and media_dir.exists():
            for img_file in media_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        img_bytes = img_file.read_bytes()
                        new_bytes = process_image_bytes(img_bytes, ocr_langs, compiled)
                        img_file.write_bytes(new_bytes)
                    except Exception:
                        pass
                count += 1
                if progress_callback:
                    percent = 100 * count / total
                    progress_callback(percent)

        # Repack DOCX
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for folder, _, files in os.walk(tmpdir):
                for fname in files:
                    full = Path(folder) / fname
                    arc = full.relative_to(tmpdir_path)
                    zout.write(str(full), str(arc))

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
        help='Example: [ {"pattern": "77-([A-Za-z0-9]{2,4})-(\\d{5,7})(?:-([A-Za-z0-9]{2,4}))?", "replacement": ""} ]\n'
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
        st.sidebar.info("Using default pattern: 77-(2–4 chars)-(5–7 chars)-optional(2–4 chars) → 4022-...")
        default_pattern = r"\b77\s*[-–—]?\s*([A-Za-z0-9]{2,4})\s*[-–—]?\s*([A-Za-z0-9]{5,7})(?:\s*[-–—]?\s*([A-Za-z0-9]{2,4}))?\b"
        def default_repl(m):
            g1 = m.group(1)
            g2 = m.group(2)
            g3 = m.group(3)
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
    Match UploadedFile lists by stem names.
    Returns list of (orig, proc), unmatched_originals, unmatched_processed.
    Matching logic:
    - If processed filename starts with 'processed_' and rest matches original stem, match.
    - Else if stems match directly, match.
    """
    matched = []
    unmatched_o = []
    unmatched_p = []
    # Build map for processed: key -> UploadedFile
    proc_map = {}
    for pf in processed_files:
        name = pf.name
        stem = Path(name).stem
        if stem.startswith("processed_"):
            orig_stem = stem[len("processed_"):]
            proc_map[orig_stem] = pf
        else:
            proc_map[stem] = pf
    for of in original_files:
        o_stem = Path(of.name).stem
        if o_stem in proc_map:
            matched.append((of, proc_map[o_stem]))
        else:
            unmatched_o.append(of)
    # Any processed not matched?
    matched_proc = set()
    for o, p in matched:
        matched_proc.add(p)
    for pf in processed_files:
        if pf not in matched_proc:
            unmatched_p.append(pf)
    return matched, unmatched_o, unmatched_p

def compare_document_pair(orig_file, proc_file):
    """
    orig_file, proc_file: UploadedFile-like with .name and .getbuffer()
    Returns dict with keys: original, processed, text_diff_html, orig_images, proc_images, error
    """
    try:
        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(orig_file.getbuffer())
            orig_path = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f2:
            f2.write(proc_file.getbuffer())
            proc_path = f2.name

        # Extract text
        orig_text = extract_text_from_docx(orig_path)
        proc_text = extract_text_from_docx(proc_path)
        text_diff_html = highlight_differences(orig_text, proc_text)

        # Extract images
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
        # Clean up temp files
        for path in ('orig_path', 'proc_path'):
            p = locals().get(path)
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

# ---------------- Main UI Pages ----------------

def single_file_page(patterns, ocr_langs, include_images):
    st.header("Single Document Processing")
    uploaded = st.file_uploader("Upload a .doc or .docx file", type=["doc", "docx"], key="single_upload")
    if not uploaded:
        st.info("Please upload a file to proceed.")
        return
    # Save uploaded to temp file
    suffix = Path(uploaded.name).suffix.lower()
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(uploaded.getbuffer())
    tmp_in_path = tmp_in.name
    tmp_in.close()

    # Convert if .doc
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

    # Preview original text
    with st.expander("📝 Original Text Preview", expanded=False):
        try:
            orig_text = extract_text_from_docx(converted_path)
        except Exception as e:
            orig_text = f"Error: {e}"
        st.text_area("Original text", orig_text, height=200)

    if st.button("▶️ Run Replacement (Single)"):
        progress_bar = st.progress(0)
        status = st.empty()
        start_time = time.time()

        # Output directory input
        output_dir = st.text_input("Output directory on server", value="output_docs")
        os.makedirs(output_dir, exist_ok=True)
        # Save under same path and name: processed_<stem>.docx
        out_name = f"processed_{Path(uploaded.name).stem}.docx"
        out_path = os.path.join(output_dir, out_name)

        status.text("🔄 Processing...")
        try:
            replace_patterns_in_docx(
                input_path=converted_path,
                output_path=out_path,
                patterns=patterns,
                ocr_langs=ocr_langs,
                include_images=include_images,
                progress_callback=lambda p: progress_bar.progress(int(p))
            )
        except Exception as e:
            st.error(f"Error: {e}")
            return
        elapsed = time.time() - start_time
        status.success(f"✅ Done in {int(elapsed)} sec")

        # Preview modified text
        with st.expander("📝 Modified Text Preview", expanded=False):
            try:
                mod_text = extract_text_from_docx(out_path)
            except Exception as e:
                mod_text = f"Error: {e}"
            st.text_area("Modified text", mod_text, height=200)

        # Diff
        status.text("🔍 Generating diff...")
        try:
            diff_html = highlight_differences(orig_text, mod_text)
        except Exception as e:
            diff_html = f"Error generating diff: {e}"
        with st.expander("🗒️ Inline Diff", expanded=True):
            st.markdown(diff_html, unsafe_allow_html=True)

        # Download and inform storage
        st.success(f"Processed file saved at: {os.path.abspath(out_path)}")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed .docx",
                data=f,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

def batch_processing_page(patterns, ocr_langs, include_images):
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
        global_progress = st.progress(0)
        global_status = st.empty()
        start_time = time.time()
        results = []
        for idx, uploaded in enumerate(uploaded_files):
            file_start = time.time()
            global_status.info(f"Processing {idx+1}/{total}: {uploaded.name}")
            # Save to temp
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpf:
                tmpf.write(uploaded.getbuffer())
                tmp_path = tmpf.name
            # Convert if .doc
            converted_path = tmp_path
            if suffix == ".doc":
                try:
                    converted = attempt_doc_to_docx(tmp_path)
                    converted_path = converted
                except Exception as e:
                    results.append({
                        'file': uploaded.name,
                        'status': f'❌ Conversion failed: {e}',
                        'time': '0',
                        'path': None
                    })
                    try: os.unlink(tmp_path)
                    except: pass
                    global_progress.progress((idx+1)/total)
                    continue
            # Output path
            out_name = f"processed_{Path(uploaded.name).stem}.docx"
            out_path = os.path.join(output_dir, out_name)
            try:
                replace_patterns_in_docx(
                    input_path=converted_path,
                    output_path=out_path,
                    patterns=patterns,
                    ocr_langs=ocr_langs,
                    include_images=include_images,
                    progress_callback=None  # no per-file progress UI here
                )
                elapsed = time.time() - file_start
                results.append({
                    'file': uploaded.name,
                    'status': '✅ Success',
                    'time': str(int(elapsed)) + "s",
                    'path': out_path
                })
            except Exception as e:
                elapsed = time.time() - file_start
                results.append({
                    'file': uploaded.name,
                    'status': f'❌ Error: {e}',
                    'time': str(int(elapsed)) + "s",
                    'path': None
                })
            finally:
                try: os.unlink(tmp_path)
                except: pass
                if converted_path != tmp_path:
                    try: os.unlink(converted_path)
                    except: pass
            global_progress.progress((idx+1)/total)
        total_elapsed = time.time() - start_time
        st.success(f"Completed {total} files in {int(total_elapsed)}s")

        # Show results and downloads
        for res in results:
            if res['status'].startswith('✅') and res['path'] and os.path.exists(res['path']):
                st.write(f"✅ {res['file']} (time: {res['time']})")
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
    The app will match by stem and show inline text diffs and image comparisons.
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
                                        st.image(res['orig_images'][i], use_column_width=True)
                                    else:
                                        st.write("No image")
                                with c2:
                                    st.text(f"Processed Image {i+1}")
                                    if i < len(res['proc_images']):
                                        st.image(res['proc_images'][i], use_column_width=True)
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
        # Manual override
        custom = st.sidebar.text_input("Tesseract executable path (Windows)", value="", help="If auto-detect fails, specify full path to tesseract.exe")
        if custom:
            if os.path.exists(custom):
                pytesseract.pytesseract.tesseract_cmd = custom
                st.sidebar.success(f"Tesseract path set to: {custom}")
            else:
                st.sidebar.error("Provided Tesseract path not found")

    # Check Tesseract
    try:
        tess_ver = pytesseract.get_tesseract_version()
        st.sidebar.success(f"Tesseract v{tess_ver} detected")
    except Exception:
        st.sidebar.error("Tesseract not found or not configured. Image OCR disabled.")

    # Navigation
    st.sidebar.divider()
    st.sidebar.title("Mode")
    mode = st.sidebar.radio("Select Mode", ["Single File", "Batch Processing", "Document Comparison"])

    # Load patterns once
    patterns = load_patterns()

    # Defaults: all OCR languages, always include images
    ocr_langs = "eng+chi_sim+chi_tra"
    include_images = True

    if mode == "Single File":
        single_file_page(patterns, ocr_langs, include_images)
    elif mode == "Batch Processing":
        batch_processing_page(patterns, ocr_langs, include_images)
    elif mode == "Document Comparison":
        comparison_page()

if __name__ == "__main__":
    main()
