import os
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import anthropic
import fitz
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import tiktoken
import pandas as pd
from datetime import datetime

# API Keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Fungsi untuk memuat FAISS index
def load_faiss_index(role, base_path="faiss"):
    faiss_index_folder = f"faiss_index_{role}"  # Misalnya 'Engineering_faiss.index'
    faiss_index_path = os.path.join(base_path, faiss_index_folder)  # Path ke folder index

    # Memuat FAISS index jika folder ada
    if os.path.exists(faiss_index_path):
        try:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.write(f"⚠️ Gagal memuat FAISS index untuk role {role}: {e}")
            return None
    else:
        st.write(f"⚠️ FAISS index untuk role {role} tidak ditemukan di {faiss_index_path}.")
        return None

# Fungsi untuk membaca PDF
def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text

# Fungsi untuk membaca DOCX
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Fungsi untuk memproses file yang di-upload dan membuat vector_store
def process_files(uploaded_files, path="faiss"):
    combined_text = ""
    file_list = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(path, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_list.append(file_name)

        if file_name.endswith(".pdf"):
            combined_text += read_pdf(file_path) + "\n"
        elif file_name.endswith(".docx"):
            combined_text += read_docx(file_path) + "\n"

    # Memecah teks menjadi chunks untuk FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_text)

    # Membuat embedding dan FAISS
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Simpan combined_text dan daftar file yang di-upload
    save_combined_text_and_files(file_list, combined_text)

    return vector_store

# Fungsi untuk menyimpan combined text dan file list
def save_combined_text_and_files(file_list, combined_text, path="faiss"):
    index_folder = f"faiss_index_temp"  # Folder sementara untuk file
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    # Menyimpan combined text ke dalam file
    with open(os.path.join(index_folder, "combine_text.txt"), "w") as f:
        f.write(combined_text)
    
    # Menyimpan daftar file yang di-upload
    with open(os.path.join(index_folder, "file_list.txt"), "w") as f:
        for file in file_list:
            f.write(file + "\n")
    
    # Simpan path folder untuk referensi selanjutnya
    return index_folder

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
    if vector_store is not None:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = ""
    
    if role == "Laws":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{prompt_laws}\n\n{context}\n\nPertanyaan: {prompt}"
    elif role == "Engineering":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{prompt_engineering}\n\n{context}\n\nPertanyaan: {prompt}"
    else:
        return "Peran tidak dikenali.", 0  # Kembalikan token_usage default

    try:
        token_usage = 0  # Default token usage

        if provider == "OpenAI":
            response = client.chat.completions.create(
                model="gpt-4o mini",
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
            return response.content, token_usage
        elif provider == "Gemini":
            model = client.GenerativeModel("gemini-pro")
            response = model.generate_content(augmented_prompt)
            return response.text, token_usage
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", 0

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
st.title("HazChat")
role = st.selectbox("Pilih Role", ["Laws", "Engineering"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Gemini"])

# Pilihan untuk load FAISS index
load_database = st.button("Load Database")
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
st.session_state.vector_store = None
# Memilih mode
if load_database:
    vector_store = load_faiss_index(role)
    if vector_store:
        st.session_state.vector_store = vector_store
        st.success("FAISS index berhasil dimuat!")
else:
    if uploaded_files:
        vector_store = process_files(uploaded_files)
        st.session_state.vector_store = vector_store
        st.success("FAISS index berhasil dibuat dari file yang di-upload!")
    else:
        st.session_state.vector_store = None

# Menyediakan download file yang di-upload
if st.session_state.vector_store:
    index_folder = f"faiss_index_temp"
    if os.path.exists(index_folder):
        st.download_button("Download Combined Text", f"{index_folder}/combine_text.txt", file_name="combine_text.txt")
        st.download_button("Download File List", f"{index_folder}/file_list.txt", file_name="file_list.txt")

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
    if client:
        response, token_usage = get_response(provider, client, prompt, role, st.session_state.vector_store, prompt_laws, prompt_engineering)
        save_to_google_sheetsQnA(prompt, response)
    else:
        response = "Provider belum diatur."
        token_usage = 0

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        if provider == "OpenAI":
            st.markdown(f"📊 Token digunakan: **{token_usage}**")
