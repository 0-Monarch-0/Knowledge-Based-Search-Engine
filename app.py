
import streamlit as st
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import textwrap

@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return embedding_model, qa_pipeline

def process_pdfs(pdf_docs, embedding_model):
    raw_text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                raw_text += page.get_text() + "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    vector_store = Chroma.from_texts(texts=chunks, embedding=embedding_model)
    return vector_store

def generate_answer(vector_store, qa_pipeline, question):
    relevant_chunks = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = f'''
    [INST]
    Use the following context to answer the question.
    Provide a concise and helpful answer based ONLY on the provided context.
    If the context does not contain the answer, say "The provided documents do not contain the answer to this question."

    Context: {context}

    Question: {question}
    [/INST]
    '''
    result = qa_pipeline(prompt, max_new_tokens=512)
    return textwrap.fill(result[0]['generated_text'].split('[/INST]')[-1].strip(), width=80)

st.set_page_config(page_title="PDF Q&A (GPU)", layout="wide")
st.title("🚀 GPU-Powered Q&A with Your Documents")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

embedding_model, qa_pipeline = load_models()

with st.sidebar:
    st.header("Upload & Process")
    pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing documents..."):
                st.session_state.vector_store = process_pdfs(pdf_docs, embedding_model)
                st.success("Processing Complete!")
        else:
            st.warning("Please upload a PDF.")
    st.markdown("---")
    st.header("New Chat")
    st.markdown("Click below to clear previous documents.")
    if st.button("Start New Chat"):
        st.session_state.vector_store = None
        st.rerun()

st.header("Ask a Question")
if st.session_state.vector_store:
    user_question = st.text_input("Ask about your documents:")
    if user_question:
        with st.spinner("Generating answer with Mistral-7B..."):
            answer = generate_answer(st.session_state.vector_store, qa_pipeline, user_question)
            st.info(answer)
else:
    st.info("Upload documents to start a new chat.")
