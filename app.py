from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.llm import LLM

if __name__ == '__main__':
    data_obj = DataProcessor(limit=5)
    chunks, data = data_obj.build_data()

    embedding = EmbeddingManager()
    model = embedding.get_model()
    chunks_list = [c.page_content for c in chunks]
    embd = embedding.embed_texts(chunks_list)

    vectordb = VectorStore()
    vectordb.add_document(data, embd)
    retriever = vectordb.get_retriever(model)

    llm = LLM(retriever)
    llm.invoke("What are the powers of the Privatisation Commission?")