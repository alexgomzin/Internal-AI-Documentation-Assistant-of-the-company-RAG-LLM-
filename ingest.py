# ingest.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import re

DATA_PATH = r"D:\company-ai-assistant\data"
DB_PATH = "data"


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents


def clean_text(text):
   
    text = re.sub(r'\n+', '\n', text)     
    text = re.sub(r'\s+', ' ', text)       
    return text.strip()

def split_documents(documents):
    # Сначала очищаем текст в каждом документе
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Теперь режем на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # уменьшил до 300 для лучшей стабильности
        chunk_overlap=30,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# 4. Создаём эмбеддинги 
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="cointegrated/rubert-tiny2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_db")  # сохраняем в папку
    return vector_db

# 5. Основной запуск
if __name__ == "__main__":
    print("Загружаем документы...")
    docs = load_documents()
    print(f"Загружено {len(docs)} документов.")
    
    print("Разбиваем на чанки...")
    chunks = split_documents(docs)
    print(f"Получено {len(chunks)} чанков.")
    
    print("Создаём векторную БД...")
    create_vector_db(chunks)
    print("Готово! База сохранена в chroma_db/")