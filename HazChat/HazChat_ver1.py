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

# Fungsi untuk memuat prompt dari folder add_prompt
def load_prompts():
    prompt_dir = os.path.join(os.getcwd(), "add_prompt")
    
    prompt_engineering_path = os.path.join(prompt_dir, "prompt_engineering.txt")
    prompt_laws_path = os.path.join(prompt_dir, "prompt_laws.txt")
    
    prompt_engineering = open(prompt_engineering_path, "r", encoding="utf-8").read() if os.path.exists(prompt_engineering_path) else None
    prompt_laws = open(prompt_laws_path, "r", encoding="utf-8").read() if os.path.exists(prompt_laws_path) else None
    
    return prompt_engineering, prompt_laws

# Memuat prompt
prompt_engineering, prompt_laws = load_prompts()

# Fungsi membaca PDF
def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text

# Fungsi membaca DOCX
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Fungsi load data dari folder yang benar
def load_knowledge(role):
    role_folders = {"Laws": "regulation", "Engineer": "engineering"}
    BASE_DIR = os.getcwd()
    data_folder = os.path.join(BASE_DIR, "data", role_folders.get(role, "data"))
    
    combined_text = ""
    if not os.path.exists(data_folder):
        st.warning(f"⚠️ Folder {data_folder} tidak ditemukan.")
        return ""
    
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            continue
        
        if file_name.endswith(".pdf"):
            combined_text += read_pdf(file_path) + "\n"
        elif file_name.endswith(".docx"):
            combined_text += read_docx(file_path) + "\n"
    
    return combined_text

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
    if not vector_store:
        return "Knowledge base kosong."
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    if role == "Laws":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n{prompt_laws}\n\nPertanyaan: {prompt}"
    elif role == "Engineer":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n{prompt_engineering}\n\nPertanyaan: {prompt}"
    else:
        return "Peran tidak dikenali."
    
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

# Streamlit UI
st.title("HazChat")
role = st.selectbox("Pilih Role", ["Laws", "Engineer"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini"])

# Load Knowledge Base
knowledge_base = load_knowledge(role)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(knowledge_base)
st.write(f"Jumlah chunks: {len(chunks)}")

# Embedding & FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embedding=embeddings) if chunks else None

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
    response = get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering) if client else "Provider belum diatur."
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
