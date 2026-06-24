#This file is responsible for reading uploaded files and converting them into plain text so the AI can process them. 
#It supports TXT, PDF, Word, CSV, Excel, and JSON files. Regardless of the original format, 
# the output is converted into text that can be used by the RAG engine and AI test case generator.


# JSON processing library
# Used for reading and formatting JSON files
import json

# Pandas library
# Used for reading CSV and Excel files
import pandas as pd

# Python DOCX library
# Used for extracting text from Word documents
from docx import Document

# PDF reader library
# Used for extracting text from PDF files
from PyPDF2 import PdfReader


# Extract text content from uploaded files
# Supports:
# - TXT
# - PDF
# - DOCX
# - CSV
# - XLSX
# - JSON
def extract_text_from_file(uploaded_file):

    # Get uploaded file name in lowercase
    # Makes extension checking case-insensitive
    file_name = uploaded_file.name.lower()

    # Process text files
    if file_name.endswith(".txt"):

        # Read file and convert bytes to string
        return uploaded_file.read().decode("utf-8")

    # Process PDF files
    elif file_name.endswith(".pdf"):

        # Create PDF reader object
        reader = PdfReader(uploaded_file)

        # Store extracted text
        text = ""

        # Loop through all pages
        for page in reader.pages:

            # Extract text from current page
            page_text = page.extract_text()

            # Add text if content exists
            if page_text:
                text += page_text + "\n"

        return text

    # Process Microsoft Word documents
    elif file_name.endswith(".docx"):

        # Load Word document
        document = Document(uploaded_file)

        # Combine all paragraphs into a single string
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    # Process CSV files
    elif file_name.endswith(".csv"):

        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)

        # Convert DataFrame to text
        return df.to_string()

    # Process Excel files
    elif file_name.endswith(".xlsx"):

        # Read Excel into DataFrame
        df = pd.read_excel(uploaded_file)

        # Convert DataFrame to text
        return df.to_string()

    # Process JSON files
    elif file_name.endswith(".json"):

        # Load JSON content
        data = json.load(uploaded_file)

        # Return formatted JSON string
        return json.dumps(
            data,
            indent=2
        )

    # Unsupported file format
    else:
        return "Unsupported file type"