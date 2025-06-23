import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import json
import re

# Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<PATH_TO_TESSERACT>'

def extract_table_data(text):
    """
    Extract component, required quantity, and storage bin from reservation items table
    """
    # Normalize text for robust matching
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    
    # Find the reservation items section
    start_idx = text.lower().find("reservation item")
    if start_idx == -1:
        return []
    
    # Extract table section - everything after "reservation item"
    table_section = text[start_idx + len("reservation item"):]
    
    # Find all components with pattern: 4digits.3digits.5digits
    components = re.findall(r'\b(\d{4}\.\d{3}\.\d{5})\b', table_section)
    
    # Find storage bins - look for all-caps words after quantity columns
    bin_pattern = r'(\b[A-Z]{3,}\b)(?!.*\b[A-Z]{3,}\b)'
    bins = re.findall(bin_pattern, table_section)
    
    # Find quantities - decimal numbers with 3 decimal places
    quantities = re.findall(r'\b(\d\.\d{3})\b', table_section)
    
    # If we found fewer quantities than components, fill with 1.000
    if len(quantities) < len(components):
        quantities = quantities + ["1.000"] * (len(components) - len(quantities))
    
    # Prepare the data
    data = []
    for i, comp in enumerate(components):
        # Get corresponding storage bin (use last found if not enough)
        bin_val = bins[i] if i < len(bins) else (bins[-1] if bins else "UNKNOWN")
        qty_val = quantities[i] if i < len(quantities) else "1.000"
        
        data.append({
            "Component": comp,
            "Req Qty": qty_val,
            "Storage Bin": bin_val
        })
    
    return data

# Streamlit UI
st.title("Production Work Order Data Extractor")
st.markdown("Upload an image or text file containing the **Reservation Items** table")

uploaded_file = st.file_uploader("Choose file", type=["jpg", "jpeg", "png", "txt"])

if uploaded_file:
    if uploaded_file.type.startswith('image'):
        # Process image with OCR
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        # Process text file
        text = uploaded_file.getvalue().decode("utf-8")
    
    # Display extracted text for debugging
    with st.expander("View Extracted Text"):
        st.text(text)
    
    # Process data
    table_data = extract_table_data(text)
    
    if table_data:
        # Show JSON output
        st.subheader("Extracted Data (JSON)")
        st.json(table_data)
        
        # Show table view
        st.subheader("Tabular View")
        df = pd.DataFrame(table_data)
        st.dataframe(df)
        
        # Show download options
        st.download_button(
            label="Download JSON",
            data=json.dumps(table_data, indent=2),
            file_name="extracted_data.json",
            mime="application/json"
        )
    else:
        st.error("No reservation items found. Check extracted text for issues.")
        st.text("Common issues:")
        st.text("- Poor image quality")
        st.text("- Component numbers not in 4.3.5 digit format")
        st.text("- No storage bins detected")

st.markdown("""
**Requirements:**
1. Install Tesseract OCR ([Installation Guide](https://github.com/tesseract-ocr/tesseract))
2. `pip install streamlit pytesseract pillow pandas`
3. Run with: `streamlit run app.py`
""")