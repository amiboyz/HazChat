import os
# from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import anthropic
import fitz  # PyMuPDF untuk PDF
from docx import Document  # untuk DOCX
import pytesseract  # OCR
from PIL import Image  # Untuk pemrosesan gambar
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
# load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Fungsi untuk membaca file PDF (dengan OCR jika diperlukan)
def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            page_text = page.get_text()
            
            # # Jika halaman tidak memiliki teks, gunakan OCR
            # if not page_text.strip():
            #     pix = page.get_pixmap()
            #     img = Image.open(io.BytesIO(pix.tobytes()))
            #     page_text = pytesseract.image_to_string(img)

            text += page_text + "\n"
    return text

# Fungsi untuk membaca file DOCX
def read_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Fungsi untuk memuat data berdasarkan role
def load_knowledge(role):
    role_folders = {
        "Laws": "data/regulation",
        "Engineer": "data/engineering"
    }
    
    data_folder = role_folders.get(role, "data")  # Default ke "data" jika role tidak ditemukan
    combined_text = ""

    if not os.path.exists(data_folder):
        return ""

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_name.endswith(".pdf"):
            combined_text += read_pdf(file_path) + "\n"
        elif file_name.endswith(".docx"):
            combined_text += read_docx(file_path) + "\n"

    return combined_text

# Streamlit UI
st.title("HazChat")

# Pilih role
role = st.selectbox("Pilih Role", ["Laws", "Engineer"])

# Pilih provider API
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini", "DeepSeek", "Llama"])

# Load data dan buat vector store berdasarkan role
knowledge_base = load_knowledge(role)

# Pisahkan teks menjadi potongan kecil
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(knowledge_base)

# Embedding dan index dengan FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embeddings)

# Fungsi untuk mengatur provider API
def set_provider(provider):
    if provider == "OpenAI":
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key)
        return client
    elif provider == "Anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(api_key=api_key)
        return client
    elif provider == "Gemini":
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        return genai
    elif provider == "DeepSeek":
        return None
    elif provider == "Llama":
        return None

# Fungsi untuk mendapatkan respons dari model
def get_response(provider, client, prompt):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\nPertanyaan: {prompt}"

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
    elif provider == "DeepSeek":
        return "DeepSeek response (belum diimplementasikan)"
    elif provider == "Llama":
        return "Llama response (belum diimplementasikan)"

# Menyimpan history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt
prompt = st.chat_input("Masukkan prompt...")

# Jika ada input, proses
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = set_provider(provider)
    if client:
        response = get_response(provider, client, prompt)
    else:
        response = f"Konfigurasi untuk {provider} belum diimplementasikan."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
