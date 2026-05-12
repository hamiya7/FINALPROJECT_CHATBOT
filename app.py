import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── HALAMAN ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Asisten Wisata Majalengka", page_icon="⛰️")
st.title("⛰️ Asisten Wisata Alam Majalengka")
st.markdown(
    "Halo! Aku siap bantu kamu merencanakan liburan, mencari rute, "
    "atau merekomendasikan tempat wisata alam seru di Majalengka. "
    "Mau ke mana kita hari ini?"
)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index_wisata",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ✅ Groq — gratis, cepat, tidak ada quota ketat
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.7,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt_template = """
Kamu adalah Asisten Wisata Lokal khusus wilayah Majalengka yang ramah, asyik,
dan sangat berpengetahuan. Gaya bahasamu santai tapi tetap sopan
(gunakan sapaan 'aku' dan 'kamu').
Gunakan informasi berikut untuk menjawab pertanyaan pengguna dengan detail.
Jika kamu tidak tahu jawabannya atau tempatnya tidak ada di informasi ini,
jujur saja bilang tidak tahu dan jangan mengarang.

Informasi Panduan:
{context}

Pertanyaan Pengguna: {question}
Jawaban Asisten:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# ── INISIALISASI ──────────────────────────────────────────────────────────────
try:
    qa = load_knowledge_base()
except Exception as e:
    st.error(f"❌ Gagal memuat knowledge base: {e}")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
if st.sidebar.button("🗑️ Hapus Riwayat Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.success("✅ Powered by Groq (Llama 3) — Fast & Free!")

# ── RIWAYAT CHAT ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── INPUT PENGGUNA ────────────────────────────────────────────────────────────
if user_input := st.chat_input("Contoh: Dimana tempat camping yang cocok buat pemula?"):

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Mencari info di buku panduan..."):
            try:
                hasil = qa.invoke({"query": user_input})
                jawaban_ai = hasil["result"]
            except Exception as e:
                jawaban_ai = f"⚠️ Terjadi kesalahan: {str(e)}"
        st.markdown(jawaban_ai)

    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})