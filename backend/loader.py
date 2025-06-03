import nltk

# Ensure NLTK's punkt tokenizer is available (used by unstructured)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace
import pdfplumber

def extract_text_with_unstructured(file_path):
    try:
        elements = partition_pdf(filename=file_path)
        return "\n\n".join(clean_extra_whitespace(el.text) for el in elements if el.text)
    except Exception as e:
        print(f"Unstructured failed: {e}")
        return None

def extract_tables_with_pdfplumber(file_path):
    try:
        text_output = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(str(cell) if cell else "" for cell in row)
                        text_output.append(row_text)
        return "\n".join(text_output)
    except Exception as e:
        print(f"PDFPlumber failed: {e}")
        return ""

def load_document(file_path):
    text = extract_text_with_unstructured(file_path)
    tables = extract_tables_with_pdfplumber(file_path)
    return f"{text}\n\n[TABLES]\n{tables}" if text else tables