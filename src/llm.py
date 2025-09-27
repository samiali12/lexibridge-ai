import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self, retriever):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                           google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

    def invoke(self, query: str):
        result = self.qa.invoke({"query": query})        
        return result
