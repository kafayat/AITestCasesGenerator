import os
import json
import pandas as pd

from docx import Document
from PyPDF2 import PdfReader


def save_uploaded_file(uploaded_file, target_folder):

    os.makedirs(target_folder, exist_ok=True)

    file_path = os.path.join(
        target_folder,
        uploaded_file.name
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def extract_text_from_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif file_name.endswith(".pdf"):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif file_name.endswith(".docx"):

        document = Document(uploaded_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    elif file_name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        return df.to_string()

    elif file_name.endswith(".xlsx"):

        df = pd.read_excel(uploaded_file)

        return df.to_string()

    elif file_name.endswith(".json"):

        data = json.load(uploaded_file)

        return json.dumps(
            data,
            indent=2
        )

    else:
        return "Unsupported file type"