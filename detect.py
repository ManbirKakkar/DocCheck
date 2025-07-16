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
    Extract all fields from reservation items table with robust matching
    """
    # Enhanced diagnostic logging
    diagnostics = []
    diagnostics.append(f"Original text length: {len(text)} characters")
    
    # Normalize text for robust matching
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    diagnostics.append(f"Normalized text length: {len(text)} characters")
    
    # More flexible header detection - handles OCR errors
    header_pattern = r'(\S{0,5}ation\s*item|item\s*list|component\s*list)'
    match = re.search(header_pattern, text, re.IGNORECASE)
    
    if match:
        diagnostics.append(f"✅ Header found: '{match.group()}' at position {match.start()}")
        table_section = text[match.end():]
        diagnostics.append(f"Table section length: {len(table_section)} characters")
        
        # Find all table rows using comprehensive pattern
        row_pattern = r'(\d{4})\s+(\d{4}\.\d{3}\.\d{5})\s+(.*?)\s+(\d+\.\d{3})\s+(\d+\.\d{3})\s*(\S*)\s+(\S+)\s+(\S+)\s+(\S+)'
        rows = re.findall(row_pattern, table_section)
        diagnostics.append(f"Found {len(rows)} rows with primary pattern")
        
        # If no rows found, try alternative pattern
        if not rows:
            alt_pattern = r'(\d{4})\s+(\d{4}\.\d{3}\.\d{5})\s+(.*?)\s+(\d+\.\d{3})\s+(\d+\.\d{3})\s+(\S*)\s+(\S+)\s+(\S+)\s+(\S+)'
            rows = re.findall(alt_pattern, table_section)
            diagnostics.append(f"Found {len(rows)} rows with alternative pattern")
        
        # Prepare the data
        data = []
        for row in rows:
            data.append({
                "Reservation Item": row[0],
                "Component": row[1],
                "Description": row[2].strip(),
                "Req Qty": row[3],
                "Comm Qty": row[4],
                "Pick Qty": row[5],
                "UoM": row[6],
                "Cd": row[7],
                "Storage Bin": row[8]
            })
        
        return data, diagnostics
    
    diagnostics.append("❌ Header not found")
    return [], diagnostics

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
    
    # Process data with diagnostics
    table_data, diagnostics = extract_table_data(text)
    
    # Show diagnostic information
    with st.expander("Diagnostic Information"):
        st.subheader("Extraction Process Details")
        for line in diagnostics:
            st.text(line)
        
        # Show component patterns found
        components = re.findall(r'\b\d{4}\.\d{3}\.\d{5}\b', text)
        st.subheader(f"Component Patterns Found: {len(components)}")
        if components:
            st.write(f"First component: {components[0]}")
        
        # Show quantities found
        quantities = re.findall(r'\b\d+\.\d{3}\b', text)
        st.subheader(f"Quantities Found: {len(quantities)}")
        if quantities:
            st.write(f"First quantity: {quantities[0]}")
        
        # Show potential headers found
        headers = re.findall(r'\S{0,5}ation\s*item|item\s*list|component\s*list', text, re.IGNORECASE)
        st.subheader(f"Potential Headers Found: {len(headers)}")
        if headers:
            st.write(f"Headers: {', '.join(headers)}")
    
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
        st.error("No reservation items found. Check diagnostic information below.")
        st.text("Common solutions:")
        st.text("- Try a higher quality image")
        st.text("- Ensure the table is visible and well-lit")
        st.text("- Check if the header appears in the extracted text")

st.markdown("""
**Requirements:**
1. Install Tesseract OCR ([Installation Guide](https://github.com/tesseract-ocr/tesseract))
2. `pip install streamlit pytesseract pillow pandas`
3. Run with: `streamlit run app.py`
""")