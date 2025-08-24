# backend/app/utils/splitter.py

import re

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def text_splitter(
    text: str,
    chunk_size: int = 10000, 
    chunk_overlap: int = 200
) -> List[str]:
    """
    Splits the input text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def text_splitter_documents(
    documents: List[str], 
    chunk_size: int = 2000, 
    chunk_overlap: int = 200
) -> List[str]:
    """
    Splits a list of documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)

def text_splitter_transcript(
    transcript_text: str, 
    transcript_id: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a transcript text into smaller chunks and returns a list of Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_text(transcript_text)
    documents = [Document(page_content=chunk, metadata={"transcript_id": transcript_id}) for chunk in chunks]
    
    return documents