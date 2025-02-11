import os
import fitz  # PyMuPDF untuk PDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import pickle

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

# Load dokumen berdasarkan role
def load_knowledge(role):
    role_folders = {"Laws": "regulation", "Engineering": "engineering"}
    BASE_DIR = os.getcwd()
    data_folder = os.path.join(BASE_DIR, "data", role_folders.get(role, "data"))

    combined_text = ""
    if not os.path.exists(data_folder):
        print(f"⚠️ Folder {data_folder} tidak ditemukan.")
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

# Proses embedding & simpan FAISS
def create_embeddings(role):
    print(f"🔄 Memproses embedding untuk role: {role}")
    knowledge_base = load_knowledge(role)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(knowledge_base)

    if chunks:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Simpan FAISS ke file
        with open(f"faiss_{role}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        
        print(f"✅ Embedding selesai! Data disimpan di faiss_{role}.pkl")
    else:
        print("⚠️ Tidak ada data untuk diproses.")

if __name__ == "__main__":
    create_embeddings("Laws")
    create_embeddings("Engineering")
