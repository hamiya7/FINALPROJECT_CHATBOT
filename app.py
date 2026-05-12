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

# Injeksi Custom CSS untuk tampilan yang lebih modern
st.markdown("""
<style>
    /* Gradient warna untuk judul */
    .title-text {
        background: -webkit-linear-gradient(45deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        padding-bottom: 0.5rem;
    }
    /* Styling untuk deskripsi/subtitle */
    .subtitle-text {
        color: #B0BEC5;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    /* Sedikit modifikasi jarak pada chat */
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header Tampilan Baru
st.markdown('<h1 class="title-text">⛰️ Asisten Wisata Alam Majalengka</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">Halo! Aku siap bantu kamu merencanakan liburan, '
    'mencari rute, atau merekomendasikan tempat wisata alam seru di Majalengka. '
    'Mau ke mana kita hari ini?</p>', 
    unsafe_allow_html=True
)
st.divider()


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

    # ✅ Groq — gratis, cepat, tidak ada quota ketat
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

# ── SIDEBAR UPGRADE ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Panel Kontrol")
    st.caption("Kelola sesi obrolanmu di sini")
    
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info("💡 **Tips:** Coba tanyakan spesifik seperti tempat camping, curug tersembunyi, atau rute hiking yang aman di Majalengka.")
    st.success("✅ Powered by Groq (Llama 3) — Fast & Free!")

# ── RIWAYAT CHAT (DENGAN CUSTOM AVATAR) ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    # Menggunakan custom icon untuk user dan asisten
    avatar_icon = "🎒" if message["role"] == "user" else "🏕️"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# ── INPUT PENGGUNA ────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ketik di sini (Contoh: Dimana tempat camping untuk pemula?)"):

    # Tampilan User
    with st.chat_message("user", avatar="🎒"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Tampilan Asisten
    with st.chat_message("assistant", avatar="🏕️"):
        with st.spinner("Mencari info di buku panduan..."):
            try:
                hasil = qa.invoke({"query": user_input})
                jawaban_ai = hasil["result"]
            except Exception as e:
                jawaban_ai = f"⚠️ Terjadi kesalahan: {str(e)}"
        st.markdown(jawaban_ai)

    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})