# backend/app/core/embeddings.py

import os
import re
import glob
import functools

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from backend.app.utils.config import GOOGLE_API_KEYS

class LawDataLoader:
    def __init__(self, base_dir: str = "backend/app/data"):
        self.base_dir = base_dir

    @functools.lru_cache(maxsize=None)
    def load_law_data(self) -> Dict[str, Dict[str, str]]:
        context = {}

        if not os.path.isdir(self.base_dir):
            print(f"Error: Base directory '{self.base_dir}' not found.")
            return context

        def load_single_file(filepath):
            filename = os.path.basename(filepath)[:-3]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return filename, f.read()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return filename, ""

        def load_topic_folder(topic_path):
            topic_name = os.path.basename(topic_path)
            file_contents = {}
            md_files = glob.glob(os.path.join(topic_path, "*.md"))

            if not md_files:
                return topic_name, {}

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = executor.map(load_single_file, md_files)
                file_contents.update(dict(results))

            return topic_name, file_contents

        topic_paths = [f.path for f in os.scandir(self.base_dir) if f.is_dir()]

        if not topic_paths:
            return context

        with ThreadPoolExecutor(max_workers=len(topic_paths) or 1) as executor:
            results = executor.map(load_topic_folder, topic_paths)
            context = dict(results)

        return context

    def get_relevant_context(self, query: str, context: Dict[str, Dict[str, str]]) -> str:
        relevant_sections = []

        for folder_title, files_dict in context.items():
            for filename, file_content in files_dict.items():
                full_title = f"{folder_title}/{filename}"
                if re.search(query, full_title, re.IGNORECASE) or re.search(query, file_content, re.IGNORECASE):
                    relevant_sections.append(f"{full_title}:{file_content}")

        return "\n\n".join(relevant_sections) if relevant_sections else "I don't know"

    def load_documents(self) -> List[Document]:
        folders_path = glob.glob(os.path.join(self.base_dir, "*"))
        text_loaders = {'autodetect_encoding': True}

        documents = []

        for folder in folders_path:
            if os.path.isdir(folder):
                loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loaders)
                docs = loader.load()
                for doc in docs:
                    filename = os.path.relpath(doc.metadata.get('source', ''), start=self.base_dir)
                    documents.append(Document(
                        page_content=doc.page_content,
                        metadata={"source": filename}
                    ))
        return documents

class VectorStore:
    def __init__(self, persist_directory: str = "backend/app/vectorstore"):
        self.persist_directory = persist_directory
        self.api_key = GOOGLE_API_KEYS[0]

        if not self.api_key:
            raise ValueError("GOOGLEAI_API_KEY environment variable is not set.")

        self.doc_embeddings = None
        self.query_embeddings = None
        self.vectorstore = None
        self.build_embeddings()

    def build_embeddings(self):
        """
        Initializes the embeddings for both query and document.
        """
        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_QUERY",
            google_api_key=self.api_key
        )   
        self.doc_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=self.api_key
        )

    def build_vectorstore(self, documents: List[Document], collection_name: str = "law_data"):
        """
        Builds and persists the vector store using the provided documents.
        """
        self.vectorstore = Chroma.from_documents(
            persist_directory=self.persist_directory,
            embedding=self.doc_embeddings,
            collection_name=collection_name,
            documents=documents
        )
        self.vectorstore.persist()
        print("Vector store persisted.")

    def load_vectorstore(self, collection_name: str = "law_data"):
        """
        Loads the vector store from the specified collection.
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"Persist directory '{self.persist_directory}' does not exist.")
        
        print(f"Loading existing collection '{collection_name}'...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=collection_name,
            embedding_function=self.doc_embeddings
        )
        print("Vector store loaded.")

    def add_transcript(self, transcript_id: str, transcript_text: str):
        """
        Adds a transcript to the vector store.
        """
        from backend.app.utils.splitter import text_splitter_transcript

        chunks = text_splitter_transcript(
            transcript_text,
            transcript_id
        )

        if not chunks:
            print(f"No chunks found for transcript ID: {transcript_id}")
            return

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        try:
            self.load_vectorstore(collection_name="transcript_data")
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading vector store: {e}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name="transcript_data",
                embedding_function=self.doc_embeddings
            )

        self.vectorstore.add_documents(chunks)

        print(f"Transcript ID {transcript_id} added to vector store.")

    def search_similar_content(self, query: str, top_k: int = 3):
        """
        Searches for similar content in vector store.
        """
        if not self.vectorstore:
            try:
                self.load_vectorstore(collection_name="transcript_data")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return []

        try:
            results = self.vectorstore.similarity_search(
                query,
                k=top_k
            )
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

# if __name__ == "__main__":
#     vectorstore = LawVectorStore()

#     vectorstore.add_transcript(
#         transcript_id="youtube_topBLaz4zgk",
#         transcript_text="Hello. Um, I'm Brodie McCloy. I'm currently a third-year medical student at Trinity College. And right now, this year, I do, uh, history and philosophy of science. So it's been a while since I've done A-level physics, chemistry, all that sort of stuff. So we'll see how this interview goes. Um, I'm gonna be showing you the sort of how we're expecting the Zoom interviews to go. So from now on, if you are not having an interview in person, you'll be invited to a Zoom interview, and right now, we're doing a natural science interview."
#     )

#     results = vectorstore.search_similar_content(
#         query="Who is Brodie McCloy?",
#         top_k=5,
#     )

#     print(results)