import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.api_core.exceptions import ResourceExhausted

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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=st.secrets["AIzaSyBXYyDShHeREuoSYdJulLUpDQc1oZpKP_A"],
        max_retries=2,          # ← batasi retry agar tidak spam
        request_timeout=30,
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

# ── RIWAYAT CHAT ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tombol hapus riwayat
if st.sidebar.button("🗑️ Hapus Riwayat Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tips:** Jika muncul error quota, tunggu sekitar 1 menit "
    "lalu coba lagi. API Gemini gratis punya batas request per menit."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── FUNGSI INVOKE DENGAN RETRY MANUAL ────────────────────────────────────────
def invoke_with_retry(qa_chain, query, max_retries=3, wait_seconds=15):
    """Coba invoke, kalau kena rate limit tunggu lalu coba lagi."""
    for attempt in range(max_retries):
        try:
            hasil = qa_chain.invoke({"query": query})
            return hasil["result"]
        except ResourceExhausted:
            if attempt < max_retries - 1:
                sisa_tunggu = wait_seconds * (attempt + 1)
                st.warning(
                    f"⏳ Quota API sedang penuh. "
                    f"Mencoba lagi dalam {sisa_tunggu} detik... "
                    f"(percobaan {attempt + 1}/{max_retries})"
                )
                time.sleep(sisa_tunggu)
            else:
                return (
                    "⚠️ Maaf, quota API Gemini sedang habis. "
                    "Silakan tunggu 1-2 menit lalu kirim pertanyaan lagi. "
                    "Ini batas gratis dari Google, bukan masalah pada aplikasi."
                )
        except Exception as e:
            return f"⚠️ Terjadi kesalahan: {str(e)}"

# ── INPUT PENGGUNA ────────────────────────────────────────────────────────────
if user_input := st.chat_input("Contoh: Dimana tempat camping yang cocok buat pemula?"):

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Mencari info di buku panduan..."):
            jawaban_ai = invoke_with_retry(qa, user_input)
        st.markdown(jawaban_ai)

    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})