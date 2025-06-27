# 📰 Research Paper Chatbot (PDF Bot)

Welcome to **PDF Bot** — an AI-powered research assistant that helps you interactively query and understand content from academic research papers (PDFs) and explore related papers from **arXiv**.

![PDF Chatbot Banner](https://img.shields.io/badge/Built%20With-LangChain%20%7C%20Streamlit%20%7C%20Groq-blueviolet?style=for-the-badge)

---

## 🚀 Features

✅ **Chat with PDFs** – Ask natural language questions about your uploaded papers  
✅ **Multi-session Chat History** – Keep your conversations organized using Session IDs  
✅ **AI-Powered Responses** – Leveraging `Groq`'s blazing-fast models (Gemma 2-9B-It)  
✅ **Context-Aware Question Reformulation** – Intelligent understanding using LangChain's retrieval augmentation  
✅ **Automatic Paper Recommendations** – Discover related research from `arXiv`  
✅ **Downloadable Chat Logs** – Export your conversations as `.txt` or `.json`  
✅ **Beautiful Streamlit UI** – Simple, intuitive, and responsive

---
## 📘 Follow the Guide (Available in Sidebar)

- ✅ **Enter a Session ID** to maintain chat history  
- 📤 **Upload one or more PDF research papers**  
- 🤖 **Ask questions in natural language**  
- 📚 **Get contextual answers** from the content  
- 🔗 **View related arXiv papers**  
- 📥 **Download the chat history** (Text / JSON)  

---

## 🧠 Under the Hood

| Component               | Description                                               |
|-------------------------|-----------------------------------------------------------|
| **LangChain**           | Core framework for chaining LLMs and tools                |
| **Groq**                | Ultra-fast LLM backend using Gemma 2 9B It                |
| **FAISS**               | In-memory vectorstore for document retrieval              |
| **HuggingFaceEmbeddings** | Embedding model for document vectorization             |
| **PyPDFLoader**         | Reads and extracts data from PDF files                    |
| **ArxivQueryRun**       | Fetches similar papers from arXiv                         |
| **ChatMessageHistory**  | Maintains session-wise conversation memory                |


---

## 🔎 Example Use Case

- **Uploaded PDF:** “Attention is All You Need”  
- **Question:** "What are the main contributions of this paper?"  
- ✅ **Answer:** Extracted directly from the context of the uploaded PDF.  
- ➕ **Suggested related papers from arXiv**
