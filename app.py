# app.py
import os
import textwrap
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

st.set_page_config(page_title="PDF Q&A (GPU)", layout="wide")
st.title("🚀 GPU-Powered Q&A with Your Documents")

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

@st.cache_resource
def load_models():
    # Embedding model (small & fast)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # LLM model - change model_name if you have memory issues
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # Quantization config (requires bitsandbytes installed)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Tokenizer & model loading
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    return embedding_model, qa_pipeline

def process_pdfs(pdf_files, embedding_model):
    raw_text = ""
    for pdf in pdf_files:
        try:
            with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
                for page in doc:
                    raw_text += page.get_text() + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    vector_store = Chroma.from_texts(texts=chunks, embedding=embedding_model)
    return vector_store

def generate_answer(vector_store, qa_pipeline, question):
    relevant_chunks = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    if not context.strip():
        return "The provided documents do not contain the answer to this question."

    prompt = f"""
[INST]
Use the following context to answer the question.
Provide a concise and helpful answer based ONLY on the provided context.
If the context does not contain the answer, say "The provided documents do not contain the answer to this question."

Context: {context}

Question: {question}
[/INST]
"""
    try:
        result = qa_pipeline(prompt, max_new_tokens=512)
        text = result[0].get('generated_text', '')
        # remove everything before final [/INST] if present
        answer = text.split('[/INST]')[-1].strip()
        return textwrap.fill(answer, width=80)
    except Exception as e:
        return f"Error generating answer: {e}"

# --- Streamlit UI ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Upload & Process")
    pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing documents..."):
                embedding_model, _ = load_models()
                st.session_state.vector_store = process_pdfs(pdf_docs, embedding_model)
                st.success("Processing Complete!")
        else:
            st.warning("Please upload at least one PDF.")

    st.markdown("---")
    st.header("New Chat")
    if st.button("Start New Chat"):
        st.session_state.vector_store = None
        st.experimental_rerun()

st.header("Ask a Question")

if st.session_state.vector_store:
    user_question = st.text_input("Ask about your documents:")
    if user_question:
        with st.spinner("Generating answer..."):
            _, qa_pipeline = load_models()
            answer = generate_answer(st.session_state.vector_store, qa_pipeline, user_question)
            st.info(answer)
else:
    st.info("Upload and process documents to start a chat.")
