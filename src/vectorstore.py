import os
from uuid import uuid4
from pathlib import Path
from typing import Any, List 
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma
from .constant import BASE_DIR


DATA_DIR = os.path.join(BASE_DIR, "data")


class VectorStore:
    def __init__(self, collection_name: str ="pakistan_laws", persist_directory = os.path.join(BASE_DIR, "data", "db")):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None 
        self.collection = None
        self.initialize_store()

    def initialize_store(self):
        try:
            dir = Path(self.persist_directory)
            dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata= {
                    "description": "Rag collection for pakistan laws"
                }
            )
            print(f"Store initialize successfully {self.collection}")
        except Exception as e:
            print(f"Error while initializing the store {e}")

    def get_len(self):
        return self.collection.count()
    
    def add_document(self, documents: List[Any], embeddings: np.array):
        ids = []
        metadatas = []
        docs_text = []
        embeds_list = []

        for i, (doc, embd) in enumerate(zip(documents, embeddings)):
            ids.append(f'doc_{uuid4().hex}')
            docs_text.append(doc.page_content)
            md = dict(doc.metadata) if getattr(doc, "metadata", None) else {}
            md.update({"doc_index": i, "content_length": len(doc.page_content)})
            metadatas.append(md)
            embeds_list.append(embd)

        self.collection.add(
            ids=ids,
            documents=docs_text,
            embeddings=embeds_list,
            metadatas=metadatas
        )
        print(f"Documents and embedding are added successfully into the collection {self.collection_name}")

    def get_retriever(self, embedding_function, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        vectorstore = Chroma(client=self.client, collection_name=self.collection_name, embedding_function=embedding_function)
        return vectorstore.as_retriever(search_kwargs=search_kwargs)