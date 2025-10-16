# 🧠 Knowledge-Based Search Engine Powered by Mistral-7B

This project transforms your static PDF documents into dynamic, conversational partners. Stop skimming through pages and start asking questions—get instant, intelligent answers synthesized by a powerful 7-billion parameter language model.

## ✨ Key Features

* **Interactive Chat Interface:** Ask questions about your documents in natural language.
* **Multiple PDF Uploads:** Upload one or more PDF files to create a unified knowledge base for your session.
* **High-Quality Answers:** Leverages the **Mistral-7B-Instruct** model for nuanced and accurate answer synthesis.
* **GPU-Accelerated:** Utilizes 4-bit quantization to run a large language model efficiently, providing fast responses.
* **Session-Based:** Your documents and questions are private to your session. Start a new chat to clear all data and begin fresh.
* **Built with Streamlit:** A clean, modern, and user-friendly interface that's fully responsive.

---

## 🛠️ Tech Stack & Architecture

This application is built on a modern, open-source stack and employs a **Retrieval-Augmented Generation (RAG)** architecture to provide answers grounded in your documents.

### Core Technologies

* **Application Framework:** Streamlit
* **Language Model (LLM):** `mistralai/Mistral-7B-Instruct-v0.2` (quantized to 4-bit)
* **Embedding Model:** `all-MiniLM-L6-v2`
* **Vector Store:** ChromaDB (in-memory)
* **Core Libraries:** LangChain, Hugging Face Transformers, PyMuPDF, PyTorch, bitsandbytes

### ⚙️ Architecture Workflow

The application follows a sequential data pipeline:

1.  **PDF Upload:** The user uploads PDF files through the Streamlit interface.
2.  **Text Extraction:** **PyMuPDF** parses the documents and extracts all raw text.
3.  **Text Chunking:** **LangChain** splits the extracted text into smaller, overlapping chunks to preserve semantic context.
4.  **Embedding & Indexing:** Each chunk is converted into a numerical vector (embedding) using the `all-MiniLM-L6-v2` model. These embeddings are then stored and indexed in a **ChromaDB** vector store.
5.  **User Query:** The user asks a question. This question is also converted into an embedding.
6.  **Semantic Search:** The vector store performs a similarity search to find the text chunks with embeddings most similar to the user's question embedding.
7.  **Prompt Augmentation:** The retrieved chunks (the "context") are combined with the user's question into a detailed prompt for the primary LLM.
8.  **Answer Generation:** The quantized **Mistral-7B** model receives the augmented prompt and generates a final, human-readable answer based *only* on the provided context.

---

## 🚀 Deployment & Inter-Process Communication (in Kaggle/Colab)

Running a web application inside a notebook environment like Kaggle or Colab requires a specific setup to expose the local server to the internet.

* **Background Process:** The Streamlit server must be launched as a background process so that the notebook can continue to run other commands. Since notebook shells often restrict backgrounding (`&`), Python's `subprocess.Popen` is used to launch the Streamlit server as a separate, non-blocking process.

* **Public Tunneling:** The Streamlit server runs locally on a specific port (e.g., `8501`) inside the Kaggle/Colab virtual machine. To make it accessible from a public browser, **pyngrok** is used. After the Streamlit server has been started, `pyngrok` initiates a secure tunnel from a public URL to the local port, effectively bridging the gap between the isolated notebook environment and the public internet.

---

## 🔮 Future Improvements

* Support for More Document Types: Add support for `.docx`, `.txt`, and even URLs.
* Chat History: Implement a feature to remember the conversation history within a session.
* Source Highlighting: Show which parts of which document were used to generate the answer.
* Alternative LLMs: Allow users to select from different language models.

---

## Acknowledgements

This project would not be possible without the incredible work of the open-source community. Special thanks to:

* **Mistral AI** for their powerful open-source models.
* **Hugging Face** for the `transformers` library and model hosting.
* **LangChain** for simplifying the development of LLM applications.
* **Streamlit** for making it easy to build beautiful data apps.

## Note
first run the model in a kaggle notebook and then access the website throught the link it provides.
