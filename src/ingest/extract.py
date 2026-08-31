import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# def extract_pdf(path: Path) -> str:
#     reader = PdfReader(str(path))
#     return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num, text))
    return pages

def extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_excel(path: Path) -> str:
    dfs = pd.read_excel(path, sheet_name=None)
    text_parts = []
    for sheet_name, df in dfs.items():
        text_parts.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
    return "\n\n".join(text_parts)

# def extract_all():
#     records = []
#     for group_folder in RAW_DIR.iterdir():
#         if not group_folder.is_dir():
#             continue
#         group_name = group_folder.name
#         for file_path in group_folder.iterdir():
#             suffix = file_path.suffix.lower()
#             try:
#                 if suffix == ".pdf":
#                     text = extract_pdf(file_path)
#                 elif suffix == ".docx":
#                     text = extract_docx(file_path)
#                 elif suffix in (".xlsx", ".xls", ".csv"):
#                     text = extract_excel(file_path)
#                 elif suffix in (".txt", ".md"):
#                     text = file_path.read_text(encoding="utf-8", errors="ignore")
#                 else:
#                     print(f"Skipping unsupported file: {file_path}")
#                     continue
#             except Exception as e:
#                 print(f"Failed to extract {file_path}: {e}")
#                 continue

#             records.append({
#                 "source_doc": file_path.name,
#                 "group": group_name,
#                 "text": text,
#             })
#             print(f"Extracted: {file_path.name} ({group_name})")

#     return records

def extract_all():
    records = []
    for group_folder in RAW_DIR.iterdir():
        if not group_folder.is_dir():
            continue
        group_name = group_folder.name
        for file_path in group_folder.iterdir():
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    pages = extract_pdf(file_path)
                    for page_num, text in pages:
                        records.append({
                            "source_doc": file_path.name,
                            "group": group_name,
                            "page": page_num,
                            "text": text,
                        })
                    print(f"Extracted: {file_path.name} ({group_name}) - {len(pages)} pages")
                    continue
                elif suffix == ".docx":
                    text = extract_docx(file_path)
                elif suffix in (".xlsx", ".xls", ".csv"):
                    text = extract_excel(file_path)
                elif suffix in (".txt", ".md"):
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    print(f"Skipping unsupported file: {file_path}")
                    continue
            except Exception as e:
                print(f"Failed to extract {file_path}: {e}")
                continue

            records.append({
                "source_doc": file_path.name,
                "group": group_name,
                "page": None,
                "text": text,
            })
            print(f"Extracted: {file_path.name} ({group_name})")

    return records

if __name__ == "__main__":
    records = extract_all()
    print(f"\nTotal documents extracted: {len(records)}")