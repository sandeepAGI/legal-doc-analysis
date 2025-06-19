import re
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def semantic_chunk(text, max_chunk_size=1000, overlap_size=200, min_chunk_size=100):
    """
    Improved semantic chunking with sentence boundaries and sliding window overlap.
    
    Args:
        text (str): Input text to chunk
        max_chunk_size (int): Target maximum size for chunks
        overlap_size (int): Character overlap between chunks
        min_chunk_size (int): Minimum chunk size to avoid tiny chunks
    
    Returns:
        List[Document]: List of chunked documents with metadata
    """
    page_pattern = re.compile(r"<PARSED TEXT FOR PAGE: (\d+) / \d+>")
    heading_pattern = re.compile(r"^(?:[IVXLCDM]+\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$|^([A-Z][A-Z\s\-]+)$")

    chunks = []
    current_section = "Introduction"
    current_page = 1

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    current_chunk_sentences = []
    current_chunk_size = 0

    for para in paragraphs:
        # Detect page number
        page_match = page_pattern.match(para)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect section headings
        if heading_pattern.match(para):
            current_section = para.strip()
            # Force chunk creation at section boundaries if we have content
            if current_chunk_sentences and current_chunk_size >= min_chunk_size:
                chunks.append(_create_chunk_document(current_chunk_sentences, current_page, current_section))
                current_chunk_sentences = []
                current_chunk_size = 0
            continue

        # Split paragraph into sentences
        sentences = nltk.sent_tokenize(para)
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max_chunk_size
            if current_chunk_size + sentence_size > max_chunk_size and current_chunk_sentences:
                # Create chunk with current sentences
                chunk = _create_chunk_document(current_chunk_sentences, current_page, current_section)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = _get_overlap_sentences(current_chunk_sentences, overlap_size)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_size = sum(len(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_size += sentence_size

    # Add any remaining sentences as a final chunk
    if current_chunk_sentences and current_chunk_size >= min_chunk_size:
        chunks.append(_create_chunk_document(current_chunk_sentences, current_page, current_section))

    return chunks


def _create_chunk_document(sentences, page, section):
    """Create a Document object from a list of sentences."""
    content = ' '.join(sentences).strip()
    return Document(
        page_content=content,
        metadata={
            "page": page,
            "section": section,
            "chunk_size": len(content),
            "sentence_count": len(sentences)
        }
    )


def _get_overlap_sentences(sentences, target_overlap_size):
    """Get sentences from the end of current chunk to create overlap."""
    if not sentences:
        return []
    
    overlap_sentences = []
    overlap_size = 0
    
    # Start from the end and work backwards
    for sentence in reversed(sentences):
        sentence_len = len(sentence)
        if overlap_size + sentence_len <= target_overlap_size:
            overlap_sentences.insert(0, sentence)
            overlap_size += sentence_len
        else:
            break
    
    return overlap_sentences


# Legacy function for backward compatibility
def semantic_chunk_legacy(text, max_chunk_size=1000):
    """Original chunking function - kept for backward compatibility."""
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