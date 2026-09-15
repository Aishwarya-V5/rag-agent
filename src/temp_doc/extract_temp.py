from pypdf import PdfReader
from docx import Document
import pandas as pd
import io


def extract_temp_file(uploaded_file):
    """
    Extracts text from an in-memory uploaded file (Streamlit's UploadedFile object).
    Returns extracted text as a single string. Nothing is saved to disk.
    """
    filename = uploaded_file.name
    suffix = filename.lower().split(".")[-1]
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file_like = io.BytesIO(file_bytes)

    if suffix == "pdf":
        reader = PdfReader(file_like)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix == "docx":
        doc = Document(file_like)
        text = "\n".join(p.text for p in doc.paragraphs)

    elif suffix in ("xlsx", "xls"):
        dfs = pd.read_excel(file_like, sheet_name=None)
        parts = []
        for sheet_name, df in dfs.items():
            parts.append(f"=== Sheet: {sheet_name} ===")
            for _, row in df.iterrows():
                row_text = " | ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
                parts.append(row_text)
        text = "\n".join(parts)

    elif suffix == "csv":
        df = pd.read_csv(file_like)
        text = df.to_string(index=False)

    elif suffix in ("txt", "md"):
        text = file_bytes.decode("utf-8", errors="ignore")

    else:
        return None, f"Unsupported file type: {suffix}"

    if not text.strip():
        return None, "No readable text extracted from this file."

    return text, None