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
    Extract all fields from reservation items table
    """
    # Normalize text for robust matching
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    
    # Find the reservation items section
    start_idx = text.lower().find("reservation item")
    if start_idx == -1:
        return []
    
    # Extract table section - everything after "reservation item"
    table_section = text[start_idx + len("reservation item"):]
    
    # Find all table rows using comprehensive pattern
    row_pattern = r'(\d{4})\s+(\d{4}\.\d{3}\.\d{5})\s+(.*?)\s+(\d+\.\d{3})\s+(\d+\.\d{3})\s*(\S*)\s+(\S+)\s+(\S+)\s+(\S+)'
    rows = re.findall(row_pattern, table_section)
    
    # Prepare the data
    data = []
    for row in rows:
        data.append({
            "Reservation Item": row[0],
            "Component": row[1],
            "Description": row[2],
            "Req Qty": row[3],
            "Comm Qty": row[4],
            "Pick Qty": row[5],
            "UoM": row[6],
            "Cd": row[7],
            "Storage Bin": row[8]
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
        st.text("- Table structure not recognized")

st.markdown("""
**Requirements:**
1. Install Tesseract OCR ([Installation Guide](https://github.com/tesseract-ocr/tesseract))
2. `pip install streamlit pytesseract pillow pandas`
3. Run with: `streamlit run app.py`
""")