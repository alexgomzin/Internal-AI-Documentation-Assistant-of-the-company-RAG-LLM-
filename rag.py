# rag.py
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.vectorstores import FAISS
MODEL_PATH = r"D:\company-ai-assistant\models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"  # ← путь к твоей модели

DB_PATH = "faiss_db"

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="cointegrated/rubert-tiny2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Загружаем LLM через llama-cpp-python
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,          # макс. длина контекста
        n_batch=512,         # сколько токенов обрабатывать за раз
        n_gpu_layers=0,      # 0 = только CPU (на Windows без CUDA)
        temperature=0.3,
        max_tokens=350,
        top_p=1,
        verbose=False,       # поставь True, чтобы видеть логи
    )
    
    # 3. Промпт
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — помощник по внутренней документации компании. Отвечай ТОЛЬКО на основе предоставленного контекста. Если в контексте нет ответа, скажи: "Я не знаю, обратитесь к HR или в поддержку. Если есть ответ, то дай овтет на русском языке"<|eot_id|><|start_header_id|>user<|end_header_id|>
Контекст:
{context}

Вопрос: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Цепочка RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain