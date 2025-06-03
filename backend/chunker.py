import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def semantic_chunk(text, max_chunk_size=1000):
    page_pattern = re.compile(r"<PARSED TEXT FOR PAGE: (\d+) / \d+>")
    heading_pattern = re.compile(r"^(?:[IVXLCDM]+\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$|^([A-Z][A-Z\s\-]+)$")

    chunks = []
    current_section = "Introduction"
    current_page = 1

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    running_text = ""
    char_count = 0

    for para in paragraphs:
        # Detect page number
        page_match = page_pattern.match(para)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect section headings
        if heading_pattern.match(para):
            current_section = para.strip()

        running_text += para + "\n"
        char_count += len(para)

        # Chunk if max size is reached
        if char_count >= max_chunk_size:
            chunks.append(Document(
                page_content=running_text.strip(),
                metadata={
                    "page": current_page,
                    "section": current_section
                }
            ))
            running_text = ""
            char_count = 0

    # Add any remaining text as a final chunk
    if running_text.strip():
        chunks.append(Document(
            page_content=running_text.strip(),
            metadata={
                "page": current_page,
                "section": current_section
            }
        ))

    return chunks