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
from streamlit_gsheets import GSheetsConnection
from datetime import datetime


# API Keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# Fungsi untuk memuat FAISS index

# Fungsi untuk memuat FAISS index dengan embeddings
def load_faiss_index(role, base_path="faiss"):
    # Tentukan path untuk folder index berdasarkan role
    faiss_index_folder = f"faiss_index_{role}"  # Misalnya 'Engineering_faiss.index'
    faiss_index_path = os.path.join(base_path, faiss_index_folder)  # Path ke folder index

    st.write(faiss_index_path)  # Menampilkan path untuk debugging

    # Memuat FAISS index jika folder ada
    if os.path.exists(faiss_index_path):
        try:
            # Membuat instance embeddings
            embeddings = OpenAIEmbeddings()

            # Memuat FAISS index menggunakan FAISS.load_local() dan memberikan embeddings
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            
            st.write(f"✅ FAISS index untuk role {role} berhasil dimuat!")
            return vector_store
        except Exception as e:
            st.write(f"⚠️ Gagal memuat FAISS index untuk role {role}: {e}")
            return None
    else:
        st.write(f"⚠️ FAISS index untuk role {role} tidak ditemukan di {faiss_index_path}.")
        return None
 
# Menghubungkan ke Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# Fungsi untuk mengambil data dari Google Sheets
def fetch_existing_data():
    # Mengambil data dari Google Sheets (Data sheet)
    existing_data = conn.read(worksheet="Context", usecols=list(range(4)),ttl=5)
    existing_data = existing_data.dropna(how="all")
    return existing_data

def fetch_existing_dataQnA():
    # Mengambil data dari Google Sheets (Data sheet)
    existing_data = conn.read(worksheet="QnA", usecols=list(range(5)),ttl=5)
    existing_data = existing_data.dropna(how="all")
    return existing_data

def save_to_google_sheets(prompt, context):
    # Mendapatkan tanggal dan jam akses
    tanggal_akses = datetime.now().strftime("%Y-%m-%d")
    jam_akses = datetime.now().strftime("%H:%M:%S")
    
    # Membuat data baru yang akan disimpan ke Google Sheets
    user_data = pd.DataFrame(
        [
            {
                "Pertanyaan": prompt,
                "build_context": context,
                "Tanggal_Akses": tanggal_akses,
                "Jam_Akses": jam_akses,
            }
        ]
    )
    # Ambil data yang sudah ada
    existing_data = fetch_existing_data()
    
    # Gabungkan data lama dengan data baru
    update_df = pd.concat([existing_data, user_data], ignore_index=True)
    
    # Update data ke Google Sheets
    conn.update(worksheet="Context", data=update_df)

def save_to_google_sheetsQnA(prompt, response):

    # Mendapatkan tanggal dan jam akses
    tanggal_akses = datetime.now().strftime("%Y-%m-%d")
    jam_akses = datetime.now().strftime("%H:%M:%S")
    
    # Membuat data baru yang akan disimpan ke Google Sheets
    user_data = pd.DataFrame(
        [
            {
                "Pertanyaan": prompt,
                "Jawaban": response,
                "Role": role,
                "Tanggal_Akses": tanggal_akses,
                "Jam_Akses": jam_akses,
            }
        ]
    )
    # Ambil data yang sudah ada
    existing_data = fetch_existing_dataQnA()
    
    # Gabungkan data lama dengan data baru
    update_df = pd.concat([existing_data, user_data], ignore_index=True)
    
    # Update data ke Google Sheets
    conn.update(worksheet="QnA", data=update_df)

# # Fungsi membaca PDF
# def read_pdf(file_path):
#     text = ""
#     with fitz.open(file_path) as pdf:
#         for page in pdf:
#             text += page.get_text() + "\n"
#     return text

# # Fungsi membaca DOCX
# def read_docx(file_path):
#     doc = Document(file_path)
#     return "\n".join([para.text for para in doc.paragraphs])

# Fungsi load data dari folder yang benar
def load_knowledge(role):
    role_folders = {"Laws": "regulation", "Engineering": "engineering"}
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
          
# Fungsi memuat prompt dari file
def load_prompts():
    prompt_dir = os.path.join(os.getcwd(), "add_prompt")

    prompt_engineering_path = os.path.join(prompt_dir, "prompt_engineering.txt")
    prompt_laws_path = os.path.join(prompt_dir, "prompt_laws.txt")
    prompt_upload_path = os.path.join(prompt_dir, "prompt_upload.txt")

    prompt_engineering = open(prompt_engineering_path, "r", encoding="utf-8").read() if os.path.exists(prompt_engineering_path) else "Tidak ada prompt engineering tersedia."
    prompt_laws = open(prompt_laws_path, "r", encoding="utf-8").read() if os.path.exists(prompt_laws_path) else "Tidak ada prompt laws tersedia."
    prompt_upload = open(prompt_upload_path, "r", encoding="utf-8").read() if os.path.exists(prompt_upload_path) else "Tidak ada prompt upload tersedia."
    return prompt_engineering, prompt_laws, prompt_upload

# **Tombol untuk melakukan embedding ulang**
# if st.button("🔄 Run Embedding"):
#     knowledge_base = load_knowledge(role)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_text(knowledge_base)
#     st.write(f"Jumlah chunks: {len(chunks)}")
#     # Embedding & FAISS
#     embeddings = OpenAIEmbeddings()
    
#     st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)

def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text() + "\n"
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file, sheet_name=None)
        for sheet_name, sheet in df.items():
            text += f"\nSheet: {sheet_name}\n"
            text += sheet.to_string(index=False) + "\n"
    return text

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
#  Buat Fungsi untuk Menghitung Token
def count_tokens(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Fungsi untuk mendapatkan respons
def get_response(provider, client, prompt, role, vector_store, prompt_laws, prompt_engineering, prompt_upload):
    # Jika FAISS tersedia, gunakan retrieval
    if vector_store is not None:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        save_to_google_sheets(prompt, context)
    else:
        context = ""
    # Jika FAISS tidak ada, hanya gunakan prompt default
    if role == "Laws":
        augmented_prompt = f"Berperanlah sebagai :\n{prompt_laws}\n jadikan data-data ini sebagai referensi dalam menjawab pertanyaan\n{context}\n\nPertanyaan: {prompt}"
    elif role == "Engineering":
        augmented_prompt = f"Berperanlah sebagai :\n{prompt_engineering}\n jadikan data-data ini sebagai referensi dalam menjawab pertanyaan\n{context}\n\nPertanyaan: {prompt}"
    elif role == "Upload":
        augmented_prompt = f"Berperanlah sebagai :\n{prompt_upload}\n jadikan data-data ini sebagai referensi dalam menjawab pertanyaan\n{context}\n\nPertanyaan: {prompt}"    
    else:
        return "Peran tidak dikenali.", 0  # Kembalikan token_usage default

    try:
        token_usage = 0  # Default token usage

        if provider == "OpenAI":
            response = client.chat.completions.create(
                model="gpt-4o",
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

# Streamlit UI
# Memuat prompt
prompt_engineering, prompt_laws, prompt_upload = load_prompts()
st.title("HazChat ")
st.markdown('Hazmi Chatbot 😎')
role = st.selectbox("Pilih Role", ["Laws", "Engineering", "Upload"])
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Gemini"])
uploaded_files = st.file_uploader("Upload file (PDF, DOCX, XLSX)", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

# Inisialisasi vector_store di awal
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# **Tombol untuk melakukan embedding ulang**
if st.button("🔄 Load Base Knowledge"):
    st.session_state.vector_store = load_faiss_index(role)

if uploaded_files:
    all_texts = []
    filenames = []
    
    for uploaded_file in uploaded_files:
        filenames.append(uploaded_file.name)
        file_text = extract_text_from_file(uploaded_file)
        all_texts.append(file_text)
    
    # Menampilkan daftar file yang diunggah
    st.write("### File yang diunggah:")
    for filename in filenames:
        st.write(f"- {filename}")
    
    # Split teks untuk embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.split_text("\n".join(all_texts))
    st.write(f"Jumlah chunks: {len(split_texts)}")
    # Hitung total token dari semua chunks
    total_tokens = sum(count_tokens(chunk) for chunk in split_texts)
    st.write(f"Total token yang digunakan: **{total_tokens}**")
    
    # Generate embeddings dan simpan ke FAISS
    embeddings = OpenAIEmbeddings()
    vector_store_upload = FAISS.from_texts(split_texts, embeddings)
    st.session_state.vector_store = vector_store_upload
    st.success("Vector store berhasil dibuat!")
    
    # Tombol untuk mengunduh teks
    all_text_combined = "\n".join(all_texts)
    st.download_button(
        label="Download Teks",
        data=all_text_combined,
        file_name="extracted_text.txt",
        mime="text/plain"
    )

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
        response, token_usage = get_response(provider, client, prompt, role, st.session_state.vector_store, prompt_laws, prompt_engineering, prompt_upload)
        save_to_google_sheetsQnA(prompt, response)
    else:
        response = "Provider belum diatur."
        token_usage = 0
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        if provider == "OpenAI":
            st.markdown(f"📊 Token digunakan: **{token_usage}**")