# app.py
import streamlit as st
from rag import get_rag_chain

st.set_page_config(page_title="AI Ассистент компании", page_icon="🤖")
st.title("🤖 Внутренний AI-ассистент")
st.caption("Задайте вопрос по HR, IT, политикам и т.д.")

# Инициализируем цепочку один раз
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain()

# История чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Выводим историю
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Получаем ответ от RAG
    with st.spinner("Думаю..."):
        response = st.session_state.rag_chain.invoke(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)