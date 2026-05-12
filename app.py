import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. KUNCI NYAWA AI (Masukkan API Key Google-mu yang asli di sini)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBXYyDShHeREuoSYdJulLUpDQc1oZpKP_A"

# 2. SETUP TAMPILAN WEB
# Menggunakan ikon gunung/tenda agar relevan dengan tema wisata alam
st.set_page_config(page_title="Asisten Wisata Majalengka", page_icon="⛰️")
st.title("⛰️ Asisten Wisata Alam Majalengka")
st.markdown("Halo! Aku siap bantu kamu merencanakan liburan, mencari rute, atau merekomendasikan tempat wisata alam seru di Majalengka. Mau ke mana kita hari ini?")

# 3. MEMUAT OTAK AI (Fungsi Cache agar tidak di-load berulang kali)
@st.cache_resource
def load_knowledge_base():
    # A. Panggil mesin penerjemah angka yang sama dengan di Colab
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # B. Load Vector Database (Catatan: allow_dangerous_deserialization wajib True di versi FAISS terbaru)
    db = FAISS.load_local("faiss_index_wisata", embeddings, allow_dangerous_deserialization=True)
    
    # C. Panggil Model Gemini (Sebagai mesin penjawab)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    # D. Atur Peran dan Gaya Bahasa (Creative Parameter)
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
    
    # E. Gabungkan menjadi satu Rantai (Chain)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}), # AI akan membaca 3 dokumen paling mirip
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Panggil fungsi otak AI-nya
qa = load_knowledge_base()

# 4. MEMORY CHATBOT (Agar chat sebelumnya tidak hilang)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. KOLOM INPUT PENGGUNA
if user_input := st.chat_input("Contoh: Dimana ya tempat camping yang cocok buat pemula?"):
    # Tampilkan teks user ke layar
    with st.chat_message("user"):
        st.markdown(user_input)
    # Simpan ke memori
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Proses jawaban AI
    with st.chat_message("assistant"):
        with st.spinner("Mencari contekan di buku panduan..."):
            # Minta jawaban ke LangChain
            hasil = qa.invoke({"query": user_input})
            jawaban_ai = hasil["result"]
            
            # Tampilkan ke layar
            st.markdown(jawaban_ai)
            
    # Simpan jawaban AI ke memori
    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})