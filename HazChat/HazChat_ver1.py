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

def load_prompts():
    try:
        # Pastikan file ada
        if os.path.exists('prompt_engineering.txt'):
            with open('prompt_engineering.txt', 'r', encoding='utf-8') as file:
                prompt_engineering = file.read()
        else:
            prompt_engineering = None
            st.warning("⚠️ File 'prompt_engineering.txt' tidak ditemukan.")
        
        if os.path.exists('prompt_laws.txt'):
            with open('prompt_laws.txt', 'r', encoding='utf-8') as file:
                prompt_laws = file.read()
        else:
            prompt_laws = None
            st.warning("⚠️ File 'prompt_laws.txt' tidak ditemukan.")

        return prompt_engineering, prompt_laws

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {str(e)}")
        return None, None

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

    # Pastikan path utama ke folder `data`
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Ambil root proyek
    data_folder = os.path.join(BASE_DIR, "data", role_folders.get(role, "data"))

    combined_text = ""
    
    # st.write(f"📂 Memuat data dari folder: {data_folder}")

    if not os.path.exists(data_folder):
        st.warning(f"⚠️ Folder {data_folder} tidak ditemukan.")
        return ""

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        
        # Pastikan hanya memproses file yang benar
        if not os.path.isfile(file_path):
            continue
        
        # st.write(f"📄 Memproses file: {file_name}")

        if file_name.endswith(".pdf"):
            text = read_pdf(file_path)
            # st.write(f"📑 Isi PDF (cuplikan): {text[:100]}")
            combined_text += text + "\n"
        elif file_name.endswith(".docx"):
            text = read_docx(file_path)
            # st.write(f"📑 Isi DOCX (cuplikan): {text[:100]}")
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
# def get_response(provider, client, prompt):
#     if not vector_store:
#         return "Knowledge base kosong."

#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     relevant_docs = retriever.get_relevant_documents(prompt)
#     context = "\n".join([doc.page_content for doc in relevant_docs])
#         if role == "Laws":
#             augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n dan {prompt_laws} Pertanyaan: {prompt}"
#         elif role == "Engineer":
#             augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n dan {prompt_engineering} Pertanyaan: {prompt}"
#     try:
#         if provider == "OpenAI":
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": augmented_prompt}]
#             )
#             return response.choices[0].message.content
#         elif provider == "Anthropic":
#             response = client.messages.create(
#                 model="claude-2",
#                 max_tokens=1024,
#                 messages=[{"role": "user", "content": augmented_prompt}]
#             )
#             return response.content
#         elif provider == "Gemini":
#             model = client.GenerativeModel("gemini-pro")
#             response = model.generate_content(augmented_prompt)
#             return response.text
#         else:
#             return f"Konfigurasi untuk {provider} belum diimplementasikan."
#     except Exception as e:
#         return f"Terjadi kesalahan: {str(e)}"
import streamlit as st

# Pastikan bahwa fungsi `set_provider` sudah didefinisikan sebelumnya.
def set_provider(provider):
    # Implementasi untuk memilih provider yang sesuai.
    # Misalnya, dapat mengembalikan klien terkait, seperti OpenAI, Anthropic, atau Gemini
    pass

def get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering):
    if not vector_store:
        return "Knowledge base kosong."

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    if role == "Laws":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n dan {prompt_laws} Pertanyaan: {prompt}"
    elif role == "Engineer":
        augmented_prompt = f"Gunakan informasi berikut jika relevan:\n{context}\n\n dan {prompt_engineering} Pertanyaan: {prompt}"
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
        else:
            return f"Konfigurasi untuk {provider} belum diimplementasikan."
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Prompt
prompt = st.chat_input("Masukkan prompt...")

# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Menetapkan provider yang digunakan
#     provider = "OpenAI"  # Tentukan provider yang diinginkan, misalnya OpenAI, Anthropic, atau Gemini
#     client = set_provider(provider)

#     # Pastikan prompt_laws dan prompt_engineering sudah didefinisikan
#     prompt_laws = "Informasi hukum terkait..."  # Sesuaikan ini dengan konteks
#     prompt_engineering = "Informasi teknik terkait..."  # Sesuaikan ini dengan konteks
#     role = "Laws"  # Tentukan role yang sesuai, bisa "Laws" atau "Engineer"

#     if client:
#         response = get_response(provider, client, prompt, role, vector_store=None, prompt_laws=prompt_laws, prompt_engineering=prompt_engineering)
#     else:
#         response = "Provider belum diatur."

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     with st.chat_message("assistant"):
#         st.markdown(response)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = set_provider(provider)
    response = get_response(provider, client, prompt) if client else "Provider belum diatur."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

