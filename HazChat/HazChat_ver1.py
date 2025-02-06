import os
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import anthropic
import fitz  # PyMuPDF untuk PDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# API Keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

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

# Fungsi untuk load data
def load_knowledge(role):
    role_folders = {"Laws": "regulation", "Engineer": "engineering"}
    data_folder = role_folders.get(role, "data")
    combined_text = ""

    st.write(f"Memuat data dari folder: {data_folder}")

    if not os.path.exists(data_folder):
        st.warning(f"Folder {data_folder} tidak ditemukan.")
        return ""

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        st.write(f"Memproses file: {file_name}")

        if file_name.endswith(".pdf"):
            text = read_pdf(file_path)
            st.write(f"Isi PDF (cuplikan): {text[:100]}")
            combined_text += text + "\n"
        elif file_name.endswith(".docx"):
            text = read_docx(file_path)
            st.write(f"Isi DOCX (cuplikan): {text[:100]}")
            combined_text += text + "\n"

    return combined_text

# Streamlit UI
st.title("HazChat")
role = st.selectbox("Pilih Role", ["Laws", "Engineer"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini", "DeepSeek", "Llama"])

# Load Knowledge Base
knowledge_base = load_knowledge(role)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(knowledge_base)
st.write(f"Jumlah chunks: {len(chunks)}")

# Embedding & FAISS
embeddings = OpenAIEmbeddings()

if chunks:
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
else:
    st.warning("Data untuk role ini kosong.")
    vector_store = None

# Fungsi untuk set provider
def set_provider(provider):
    if provider == "OpenAI":
        return OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "Anthropic":
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "Gemini":
        genai.configure(api_key=GEMINI_API_KEY)
        return genai
    else:
        return None

# Fungsi untuk mendapatkan respons
def get_response(provider, client, prompt):
    if not vector_store:
        return "Knowledge base kosong."

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\nPertanyaan: {prompt}"

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
        else:
            return f"Konfigurasi untuk {provider} belum diimplementasikan."
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Prompt
prompt = st.chat_input("Masukkan prompt...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = set_provider(provider)
    response = get_response(provider, client, prompt) if client else "Provider belum diatur."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
