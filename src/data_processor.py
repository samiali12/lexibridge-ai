import re 
import os
import json 

from dotenv import load_dotenv

from .constant import BASE_DIR

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DATA_PATH = os.path.join(BASE_DIR, "data", "pakistan_laws.json")


class DataProcessor:
    def __init__(self, data_path=DATA_PATH, limit=50):
        self.data_path = data_path
        self.limit = limit

    def _data_loader(self):
        with open(DATA_PATH, 'r', encoding="utf-8") as file:
            data = json.load(file)
        return data


    def _convert_unicode_escape_to_text(self, text):
        if not isinstance(text, str):
            return text 
    
        try:
            return text.encode("utf-8").decode("unicode-escape")
        except Exception as e:
            return text
    

    def _remove_headers_footers(self, text):
        t = re.sub(r'Page\s*\d+\s*of\s*\d+', ' ', text, flags=re.IGNORECASE)
        t = "\n".join([line for line in t.splitlines() if len(line.strip())>2 or line.strip().endswith('.')])
        t = re.sub(r'\n{2,}', '\n\n', t).strip()
        return t 


    def _preprocess(self,data):
        cleaned_data = []
        for entry in data:
            raw = entry['text']
            raw = self._convert_unicode_escape_to_text(raw)
            raw = self._remove_headers_footers(raw)
            temp_data = {
                'file_name': entry['file_name'],
                'text': raw
            }
            cleaned_data.append(temp_data)
        cleaned_data = cleaned_data[:self.limit]
        return cleaned_data


    def _load_data(self):
        data = self._data_loader()
        data = self._preprocess(data)
        documents = [ Document(page_content=entry['text'], metadata={'source': entry['file_name']}) for entry in data]
        return documents

    def _chunk_data(self, data):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
        data = text_splitter.split_documents(data)
        return data


    def build_data(self):
        data = self._load_data()
        chunks = self._chunk_data(data)
        return chunks, data
