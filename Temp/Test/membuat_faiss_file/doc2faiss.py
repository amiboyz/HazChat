import os
import zipfile  # untuk membuat file ZIP
import fitz  # PyMuPDF untuk membaca PDF
from docx import Document  # Untuk membaca file DOCX
import tiktoken  # Untuk menghitung token
import io
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Memuat API Key dari .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Direktori penyimpanan FAISS Index
FAISS_PATH = "Output_Faiss"

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

# Fungsi menghitung token
def count_tokens(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Fungsi menyimpan FAISS ke file dan menyimpan daftar nama file dalam notepad
def save_faiss_index(vector_store, file_list, path=FAISS_PATH):
    index_folder = f"{path}/faiss_index"

    # Membuat folder untuk menyimpan hasil FAISS jika belum ada
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    # Simpan FAISS index
    vector_store.save_local(index_folder)
    print(f"✅ FAISS index telah disimpan di: {index_folder}")

    # Simpan daftar file yang digunakan ke dalam file_list.txt
    file_list_path = os.path.join(index_folder, "file_list.txt")
    with open(file_list_path, "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

    print(f"✅ Daftar file telah disimpan di: {file_list_path}")

    # Membuat file ZIP dari hasil FAISS
    zip_filename = f"{index_folder}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Menambahkan file FAISS dan file_list.txt ke dalam ZIP
        for foldername, subfolders, filenames in os.walk(index_folder):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, index_folder))

    print(f"✅ File FAISS index telah dikemas ke dalam zip: {zip_filename}")
    return zip_filename

# Fungsi untuk memproses file yang di-upload
def process_files(uploaded_files, path=FAISS_PATH):
    combined_text = ""
    file_list = []

    # Membuat folder sementara jika belum ada
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")

    # Simpan file yang diupload dan ekstrak teks
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join("temp_files", file_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_list.append(file_name)

        # Proses file berdasarkan ekstensi
        if file_name.endswith(".pdf"):
            combined_text += read_pdf(file_path) + "\n"

    # Memecah teks menjadi chunks untuk FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_text)

    # Menghitung total token
    total_tokens = sum(count_tokens(chunk) for chunk in chunks)
    print(f"Total token yang digunakan: {total_tokens}")

    # Membuat embedding dan FAISS
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Simpan FAISS index dalam bentuk ZIP
    zip_file = save_faiss_index(vector_store, file_list)

    # Simpan combined_text ke dalam file combine_text.txt
    index_folder = f"{path}/faiss_index"
    file_list_path = os.path.join(index_folder, "combine_text.txt")
    with open(file_list_path, "w") as f:
        f.write(combined_text)  # Simpan seluruh combined_text ke dalam file

    return zip_file

# Streamlit Interface
def main():
    st.title("FAISS Index Generator")

    # Upload file PDF dan DOCX
    uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        # Proses file yang di-upload dan buat FAISS index
        with st.spinner("Membuat FAISS index..."):
            zip_file = process_files(uploaded_files)

        # Berikan link untuk download file ZIP
        st.success("FAISS index berhasil dibuat!")
        st.download_button("Download FAISS Index (ZIP)", zip_file, file_name="faiss_index.zip", mime="application/zip")

main()
