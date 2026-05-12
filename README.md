# ⛰️ Asisten Wisata Alam Majalengka

Chatbot cerdas berbasis AI yang dirancang untuk membantu wisatawan menemukan destinasi alam terbaik di Kabupaten Majalengka. Proyek ini menggunakan arsitektur **RAG (Retrieval-Augmented Generation)** untuk memberikan jawaban yang akurat berdasarkan data wisata lokal yang spesifik.

🔗 **Live Demo**: [finalproject-chatbot.streamlit.app](https://finalproject-chatbot.streamlit.app)

---

## 🚀 Fitur Utama

- 🗺️ **Pemandu Lokal Virtual** — Memberikan rekomendasi destinasi (Curug, Terasering, Camping Ground) dengan gaya bahasa yang ramah dan santai.
- 🧠 **Teknologi RAG** — Menggabungkan kekuatan **Groq (Llama 3.1)** dengan basis data pengetahuan lokal menggunakan **FAISS**.
- 💬 **Memory Chat** — Menyimpan riwayat percakapan agar interaksi terasa lebih alami.
- 🌐 **Antarmuka Streamlit** — Tampilan web yang responsif dan mudah digunakan.

---

## 🛠️ Arsitektur Teknologi

| Komponen | Teknologi |
|---|---|
| Framework | LangChain |
| LLM | Groq — `llama-3.1-8b-instant` |
| Vector Database | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Web App | Streamlit |

---

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/hamiya7/FINALPROJECT_CHATBOT.git
cd FINALPROJECT_CHATBOT
```

### 2. Install Dependensi
Pastikan menggunakan **Python 3.11**, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi API Key
Buat file `.streamlit/secrets.toml` dan isi dengan:
```toml
GROQ_API_KEY = "isi_api_key_groq_kamu"
```
> Dapatkan API key gratis di [console.groq.com](https://console.groq.com)

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

---

## 📂 Struktur Folder

```
FINALPROJECT_CHATBOT/
│
├── faiss_index_wisata/     # Database vektor (index pengetahuan wisata)
├── app.py                  # File utama aplikasi Streamlit
├── requirements.txt        # Daftar library yang dibutuhkan
└── README.md               # Dokumentasi proyek
```

---

## 📋 Requirements

```
streamlit
langchain>=0.2.0
langchain-community>=0.2.0
langchain-groq>=0.1.0
faiss-cpu
sentence-transformers
```

---

## 👤 Pengembang

**Hamiya Aisya Mardhiya**  
Mahasiswa D4 Teknologi Rekayasa Informatika Industri  
Politeknik Manufaktur Bandung — Kelas 3AEC3  

> Proyek ini dibuat sebagai bagian dari **Final Project** pengembangan sistem chatbot berbasis AI.
