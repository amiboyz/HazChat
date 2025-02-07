import os
import streamlit as st
import pickle
from openai import OpenAI
import google.generativeai as genai
import anthropic
from langchain_community.vectorstores import FAISS

# API Keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Fungsi untuk memuat FAISS yang sudah ada
def load_faiss(role):
    file_path = f"faiss_{role}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        return None

# Fungsi untuk menjalankan precompute_embeddings.py
def run_precompute_embeddings():
    os.system("python precompute_embeddings.py")
    st.success("✅ Embedding selesai! Silakan refresh halaman.")

# Streamlit UI
st.title("HazChat")
role = st.selectbox("Pilih Role", ["Laws", "Engineering"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini"])

# **Tombol untuk melakukan embedding ulang**
if st.button("🔄 Jalankan Embedding (Jika Ada Data Baru)"):
    run_precompute_embeddings()

# Load FAISS
vector_store = load_faiss(role)
if vector_store:
    st.success(f"✅ Knowledge base untuk {role} berhasil dimuat!")
else:
    st.warning(f"⚠️ Tidak ada knowledge base untuk {role}. Klik tombol di atas untuk membuat embedding.")

# Fungsi untuk memilih provider
def set_provider(provider):
    if provider == "OpenAI":
        return OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "Anthropic":
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "Gemini":
        genai.configure(api_key=GEMINI_API_KEY)
        return genai
    return None

# Fungsi untuk mendapatkan respons
def get_response(provider, client, prompt, role, vector_store):
    if not vector_store:
        return "Knowledge base kosong. Silakan buat embedding dulu."

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    augmented_prompt = f"Gunakan informasi berikut:\n{context}\n\nPertanyaan: {prompt}"

    try:
        if provider == "OpenAI":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": augmented_prompt}]
            )
            return response.choices[0].message.content
        elif provider == "Anthropic":
            response = client.messages.create(
                model="claude-2",
                max_tokens=1024,
                messages=[{"role": "user", "content": augmented_prompt}]
            )
            return response.content
        elif provider == "Gemini":
            model = client.GenerativeModel("gemini-pro")
            response = model.generate_content(augmented_prompt)
            return response.text
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# Chat Input
prompt = st.chat_input("Masukkan prompt...")
if prompt:
    client = set_provider(provider)
    response = get_response(provider, client, prompt, role, vector_store) if client else "Provider belum diatur."
    
    st.chat_message("user").markdown(prompt)
    st.chat_message("assistant").markdown(response)
