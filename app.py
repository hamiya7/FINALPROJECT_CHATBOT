import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Asisten Wisata Majalengka", page_icon="⛰️")
st.title("⛰️ Asisten Wisata Alam Majalengka")
st.markdown("Halo! Aku siap bantu kamu merencanakan liburan, mencari rute, atau merekomendasikan tempat wisata alam seru di Majalengka. Mau ke mana kita hari ini?")

@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index_wisata", embeddings, allow_dangerous_deserialization=True)
    
    # ✅ PERBAIKAN 1: Ganti model ke gemini-2.0-flash yang masih aktif
    # ✅ PERBAIKAN 2: Ambil API key dari st.secrets (wajib di Streamlit Cloud)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # <-- diganti dari gemini-1.5-flash
        temperature=0.7,
        google_api_key=st.secrets["GOOGLE_API_KEY"]  # <-- tambahkan ini
    )
    
    prompt_template = """
    Kamu adalah Asisten Wisata Lokal khusus wilayah Majalengka yang ramah, asyik, dan sangat berpengetahuan. 
    Gaya bahasamu santai tapi tetap sopan (gunakan sapaan 'aku' dan 'kamu').
    Gunakan informasi berikut untuk menjawab pertanyaan pengguna dengan detail.
    Jika kamu tidak tahu jawabannya atau tempatnya tidak ada di informasi ini, jujur saja bilang tidak tahu dan jangan mengarang.

    Informasi Panduan:
    {context}

    Pertanyaan Pengguna: {question}
    Jawaban Asisten:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa = load_knowledge_base()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Contoh: Dimana ya tempat camping yang cocok buat pemula?"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Mencari contekan di buku panduan..."):
            hasil = qa.invoke({"query": user_input})
            jawaban_ai = hasil["result"]
            st.markdown(jawaban_ai)
            
    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})