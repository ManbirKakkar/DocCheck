import streamlit as st
import pytesseract
import re
from PIL import Image
import cv2
import numpy as np
import json

# Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<your_tesseract_path>'

st.title("🔍 OCR Data Extractor")
st.subheader("Upload an image to extract MAT, S/N, and BID")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Enhanced image preprocessing for OCR"""
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 5
    )
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Apply dilation to connect broken text
    kernel = np.ones((2, 1), np.uint8)  # Vertical kernel
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def extract_data(image):
    """Extract MAT, S/N, and BID using OCR with enhanced patterns"""
    text = pytesseract.image_to_string(image)
    
    # Define patterns with priority
    patterns = {
        "MAT": [
            r"Mat[^:\n]*[:]?[^\d]*(\d{4}\.\d+\.\d+)",  # Primary pattern
            r"\b(\d{4}\.\d{3}\.\d{5})\b"  # Fallback: direct number pattern
        ],
        "S/N": [
            r"(?:S[\\/I]N|S/N)[^:\n]*[:]?[^\d]*(\d+)",
            r"\bS[^:\d]*(\d{9,})\b"  # Fallback: digits after S-like pattern
        ],
        "BID": [
            # Handle colon, hyphen, space or no separator
            r"B[Ii][DdE][^:\d\n]*[:]?[-]?[^\d]*(\d{9,})",  
            r"\bB[^:\d]*(\d{9,})\b",  # Fallback: digits after B-like pattern
            r"(\d{15,})"  # Final fallback: very long digit sequences
        ]
    }
    
    results = {}
    
    # Try each pattern in priority order
    for field, pattern_list in patterns.items():
        value = None
        for pattern in pattern_list:
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                # For BID, verify it's long enough
                if field == "BID" and len(value) < 9:
                    continue
                break
        results[field] = value or "Not found"
    
    return results, text

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Extract data
    extracted, raw_text = extract_data(processed_img)
    
    # Create JSON output
    json_output = json.dumps(extracted, indent=2)
    
    st.subheader("Extracted Data (JSON Format):")
    st.json(extracted)
    
    # Download JSON button
    st.download_button(
        label="Download as JSON",
        data=json_output,
        file_name="extracted_data.json",
        mime="application/json"
    )
    
    with st.expander("View OCR Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            st.image(processed_img, use_column_width=True, clamp=True)
        
        st.subheader("Raw OCR Text")
        st.code(raw_text)
        
        st.subheader("Pattern Debugging")
        st.write(f"MAT Pattern Used: {extracted.get('MAT_DEBUG', 'N/A')}")
        st.write(f"S/N Pattern Used: {extracted.get('SN_DEBUG', 'N/A')}")
        st.write(f"BID Pattern Used: {extracted.get('BID_DEBUG', 'N/A')}")

st.markdown("---")
st.caption("Note: Handles hyphens, colons, and other separators in BID labels")