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
    
    # Debug: Periksa apakah file ada
    st.write(f"Memeriksa keberadaan file: {file_path}")
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.write("📂 FAISS berhasil dimuat!")
            return pickle.load(f)
    else:
        st.warning(f"⚠️ FAISS tidak ditemukan untuk {role}.")
        return None

# Fungsi untuk menjalankan precompute_embeddings.py
def run_precompute_embeddings():
    try:
        # Menjalankan skrip precompute_embeddings.py menggunakan subprocess
        result = subprocess.run(["python", "precompute_embeddings.py"], check=True, capture_output=True, text=True)
        st.success("✅ Embedding selesai!")
        st.write(f"Output: {result.stdout}")
    except Exception as e:
        # Menangani kesalahan jika ada masalah saat menjalankan skrip
        st.error(f"❌ Terjadi kesalahan saat menjalankan precompute_embeddings.py: {str(e)}")
        if e.stdout:
            st.write(f"stdout: {e.stdout}")
        if e.stderr:
            st.write(f"stderr: {e.stderr}")

# Fungsi memuat prompt dari file
def load_prompts():
    prompt_dir = os.path.join(os.getcwd(), "add_prompt")

    prompt_engineering_path = os.path.join(prompt_dir, "prompt_engineering.txt")
    prompt_laws_path = os.path.join(prompt_dir, "prompt_laws.txt")

    prompt_engineering = open(prompt_engineering_path, "r", encoding="utf-8").read() if os.path.exists(prompt_engineering_path) else "Tidak ada prompt engineering tersedia."
    prompt_laws = open(prompt_laws_path, "r", encoding="utf-8").read() if os.path.exists(prompt_laws_path) else "Tidak ada prompt laws tersedia."

    return prompt_engineering, prompt_laws

# Memuat prompt
prompt_engineering, prompt_laws = load_prompts()

# Streamlit UI
st.title("HazChat (Hazmi Chatbot)")
role = st.selectbox("Pilih Role", ["Laws", "Engineering"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini"])

# **Tombol untuk melakukan embedding ulang**
if st.button("🔄 Jalankan Embedding"):
    run_precompute_embeddings()

# Load FAISS
vector_store = load_faiss(role)
if vector_store:
    st.success(f"✅ Knowledge base untuk {role} berhasil dimuat!")
else:
    st.warning(f"⚠️ Tidak ada knowledge base untuk {role}. Chatbot tetap bisa berjalan hanya dengan prompt bawaan.")

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
def get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering):
    # Jika FAISS tersedia, gunakan retrieval
    if vector_store:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = ""

    # Jika FAISS tidak ada, hanya gunakan prompt default
    if role == "Laws":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{prompt_laws}\n\n{context}\n\nPertanyaan: {prompt}"
    elif role == "Engineering":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{prompt_engineering}\n\n{context}\n\nPertanyaan: {prompt}"
    else:
        return "Peran tidak dikenali."

    try:
        token_usage = 0  # Default token usage

        if provider == "OpenAI":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": augmented_prompt}]
            )
            token_usage = response.usage.total_tokens  # OpenAI API memberikan jumlah token
            return response.choices[0].message.content, token_usage
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
    
# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Masukkan prompt...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    client = set_provider(provider)
    if provider == "OpenAI":
        response, token_usage = get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering) if client else "Provider belum diatur."
    if provider == "Gemini":
        response = get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering) if client else "Provider belum diatur."
   
    st.session_state.messages.append({"role": "assistant", "content": response})
    if provider == "OpenAI":
        with st.chat_message("assistant"):
            st.markdown(response)
            st.info(f"📊 Token digunakan: **{token_usage}**")
    if provider == "Gemini":
        with st.chat_message("assistant"):
            st.markdown(response)
        