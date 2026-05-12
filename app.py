import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── HALAMAN & CUSTOM CSS ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asisten Wisata Majalengka", 
    page_icon="⛰️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Injeksi Custom CSS untuk mengubah seluruh layout menjadi SATU CARD WIDGET
st.markdown("""
<style>
    /* 1. Latar belakang luar aplikasi (Dibuat sangat gelap agar Card Chat menonjol) */
    [data-testid="stAppViewContainer"] {
        background-color: #0d1117; 
    }
    
    /* 2. Mengubah Container Utama Streamlit menjadi SATU CARD WIDGET */
    .block-container {
        background-color: #161b22; /* Warna background dalam chat */
        border-radius: 24px; /* Sudut membulat ala widget */
        border: 1px solid #2ea043; /* Border garis tepi hijau */
        box-shadow: 0 12px 40px rgba(0,0,0,0.5); /* Efek bayangan melayang */
        max-width: 750px;
        padding-top: 0rem !important; /* Hilangkan padding atas */
        padding-bottom: 2rem !important;
        margin-top: 3rem;
        overflow: hidden; /* Supaya header menempel sempurna di sudut */
    }

    /* 3. Styling Header di dalam Card (Nempel ke ujung atas seperti Tidio) */
    .widget-header {
        background: linear-gradient(135deg, #34a853, #238636); /* Gradasi Hijau */
        margin: 0 -4rem 2rem -4rem; /* Menarik background sampai ke tepi container */
        padding: 2.5rem 2rem;
        text-align: center;
        border-bottom: 3px solid #1e6b2a;
    }
    
    .title-text {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #e6f4ea;
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
    }

    /* 4. Menghilangkan kotak default Streamlit pada pesan chat */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 5px 0 !important;
    }

    /* 5. Membuat teks pesan menjadi "Bubble" di dalam Card utama */
    div[data-testid="stChatMessage"] .stMarkdown {
        background-color: #21262d; /* Warna bubble */
        padding: 12px 18px;
        border-radius: 4px 16px 16px 16px; /* Bentuk lengkung bubble chat */
        border: 1px solid #30363d;
        display: inline-block;
    }

    /* 6. Custom Tombol Sidebar */
    div.stButton > button:first-child {
        background-color: rgba(52, 168, 83, 0.1);
        color: #34a853; 
        border: 1px solid #34a853;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #34a853; 
        color: white;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER WIDGET (Bagian atas dari Jendela Chat) ─────────────────────────────
st.markdown("""
<div class="widget-header">
    <div class="title-text">⛰️ Asisten Wisata Majalengka</div>
    <div class="subtitle-text">Halo! Aku siap bantu kamu merencanakan liburan, mencari rute, atau merekomendasikan tempat wisata alam seru di Majalengka. Mau ke mana kita hari ini?</div>
</div>
""", unsafe_allow_html=True)


# ── LOAD MODEL (TIDAK ADA PERUBAHAN) ──────────────────────────────────────────
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

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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
with st.sidebar:
    st.markdown("### ⚙️ Panel Kontrol")
    st.caption("Kelola sesi obrolanmu di sini")
    
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info("💡 **Tips:** Coba tanyakan spesifik seperti tempat camping, curug tersembunyi, atau rute hiking yang aman di Majalengka.")
    st.success("✅ Powered by Groq (Llama 3) — Fast & Free!")

# ── RIWAYAT CHAT ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar_icon = "🎒" if message["role"] == "user" else "⛰️"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# ── INPUT PENGGUNA ────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ketik di sini (Contoh: Dimana tempat wisata yang sejuk?)"):

    with st.chat_message("user", avatar="🎒"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="⛰️"):
        with st.spinner("Mencari info di buku panduan..."):
            try:
                hasil = qa.invoke({"query": user_input})
                jawaban_ai = hasil["result"]
            except Exception as e:
                jawaban_ai = f"⚠️ Terjadi kesalahan: {str(e)}"
        st.markdown(jawaban_ai)

    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})