# backend/app/utils/splitter.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_splitter(text: str, chunk_size: int = 10000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits the input text into smaller chunks for processing.
    
    Args:
        text (str): The text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters that overlap between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def text_splitter_documents(documents: List[str], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits a list of documents into smaller chunks.
    
    Args:
        documents (List[str]): The list of documents to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters that overlap between chunks.
    
    Returns:
        List[str]: A list of text chunks from all documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)