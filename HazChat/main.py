import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import anthropic

# Load environment variables
load_dotenv()

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
        # Tambahkan konfigurasi DeepSeek jika tersedia
        return None
    elif provider == "Llama":
        # Tambahkan konfigurasi Llama jika tersedia
        return None

# Fungsi untuk mendapatkan respons dari model
def get_response(provider, client, prompt):
    if provider == "OpenAI":
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Ganti dengan model yang diinginkan
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    elif provider == "Anthropic":
        response = client.messages.create(
            model="claude-2",  # Ganti dengan model yang diinginkan
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content
    elif provider == "Gemini":
        model = client.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    elif provider == "DeepSeek":
        return "DeepSeek response (belum diimplementasikan)"
    elif provider == "Llama":
        return "Llama response (belum diimplementasikan)"

# Antarmuka Streamlit
st.title("Private LLM dengan Streamlit")

# Pilih provider API
provider = st.selectbox("Pilih Provider API", ["OpenAI", "Anthropic", "Gemini", "DeepSeek", "Llama"])

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
    # Tambahkan pesan user ke state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Dapatkan response dari model
    client = set_provider(provider)
    if client:
        response = get_response(provider, client, prompt)
    else:
        response = f"Konfigurasi untuk {provider} belum diimplementasikan."
    
    # Tambahkan respons ke state
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)