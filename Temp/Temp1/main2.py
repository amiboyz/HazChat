import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Streamlit UI
st.title("📄 Private LLM with PDF Integration")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=client.ChatCompletion,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Input prompt
    prompt = st.text_area("Masukkan Pertanyaan Berdasarkan PDF")

    if st.button("Kirim") and prompt:
        response = qa.run(prompt)
        st.write("### 📢 Jawaban:")
        st.write(response)

    # Cleanup temporary file
    os.remove("temp.pdf")
