import fitz  # PyMuPDF
import re
import glob  # Global file searching
from database import db  # Importing database for storing chunks (modify if necessary)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Specify the path to the PDF files (adjust as needed)
pdf_files = glob.glob(r"C:\SourceCodeOpenAIBot\data\FAQ2.pdf")
if not pdf_files:
    raise FileNotFoundError("No PDF files found at the specified path.")
pdf_path = pdf_files[0]

def extract_text_from_pdf(pdf_path, remove_all_punctuation=False):
    """
    Extract text from a PDF file and clean it.
    
    Parameters:
    pdf_path (str): Path to the PDF file.
    remove_all_punctuation (bool): If True, removes all punctuation; if False, retains useful punctuation.
    
    Returns:
    str: Cleaned text extracted from the PDF.
    """
    try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)
        full_text = ""
        
        # Extract text from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text")
            full_text += page_text + " "
        
        pdf_document.close()
        
        # Clean extracted text
        full_text = re.sub(r'\s+', ' ', full_text).strip()  # Normalize whitespace
        full_text = re.sub(r'http[s]?://\S+|www\.\S+', '', full_text)  # Remove URLs
        full_text = re.sub(r'CITATIONS \d+ READS \d+', '', full_text)  # Remove citation stats
        full_text = re.sub(r'All content following this page.*?uploaded by .*? on \d+ \w+ \d+', '', full_text, flags=re.DOTALL)
        full_text = re.sub(r'© .*? All Rights Reserved.*?\b', '', full_text, flags=re.DOTALL)  # Remove copyright notices
        full_text = re.sub(r'((\d+[,\s]*){3,})|((\d+\.\d+\s*){3,})', '', full_text)  # Remove table-like structures

        # Optional punctuation removal
        if remove_all_punctuation:
            full_text = re.sub(r'[^\w\s]', '', full_text)
        else:
            full_text = re.sub(r'[^\w\s\.\?\!]', '', full_text)

        return full_text.lower()  # Convert text to lowercase
    except Exception as e:
        print(f"An error occurred during PDF text extraction: {e}")
        return ""


def chunk_text(text, max_chunk_size=1000, chunk_overlap=200):
    """
    Split the text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Parameters:
    text (str): The input text to be split.
    max_chunk_size (int): The maximum number of characters for each chunk.
    chunk_overlap (int): The number of characters to overlap between chunks.
    
    Returns:
    list: A list of text chunks.
    """

    # Define the separators in order of importance
    separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    # RecursiveTextSplitter is used here because the splitting is handled internally by the text splitter
    # with its parameters:  it processes the entire text and returns a list of chunks
    # All the logic of when to start a new chunk, how to handle overlaps, 
    # and how to use separators is handled by the text splitter internally.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = text_splitter.split_text(text)

    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c}\n---")

    return chunks



"""def chunk_text(text, chunk_size=300):
    
    Split the text into smaller chunks of a specified size.
    
    Parameters:
    text (str): The input text to be split.
    chunk_size (int): The maximum number of characters for each chunk.
    
    Returns:
    list: A list of text chunks.
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in text.split():
        if current_size + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word) + 1
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    # Add any remaining text as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
"""

def process_pdf_and_store(pdf_path):
    """
    Extract text from a PDF, split it into chunks, and store them in the database.
    """
    # Extract text from the PDF
    full_text = extract_text_from_pdf(pdf_path, remove_all_punctuation=False)
    
    # Chunk the extracted text
    chunks = chunk_text(full_text)
    
    # Store the chunks in the database
    if chunks:
        filename = pdf_path.split('\\')[-1]  # Extract the filename from the path
        db.insert_chunks_and_embeddings(filename, chunks)  # Ensure this method is defined in your 'db' module
        print(f"Processed and stored {len(chunks)} high-quality chunks from {filename}")
    else:
        print(f"No high-quality chunks found in {pdf_path}")

    return chunks

# Process the PDF and store its chunks
processed_chunks = process_pdf_and_store(pdf_path)
print(processed_chunks) 

if processed_chunks:
    print(f"Successfully processed and stored {len(processed_chunks)} chunks from {pdf_path}")
else:
    print("No chunks were processed or stored.")