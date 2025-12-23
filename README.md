# RAG Document Q&A with Groq & LangChain

This repository contains a Streamlit application that allows users to chat with their PDF documents. It uses **RAG (Retrieval-Augmented Generation)** to fetch relevant content from uploaded research papers and uses **Groq's** high-speed inference engine to generate accurate answers.

## 🛠️ Environment Setup

To run this application, you need to set up your environment and install the necessary dependencies.

### 1. Create a Virtual Environment

```bash
conda create -p venv python==3.10 -y
conda activate venv/

```

### 2. Install Dependencies

You need several libraries for the UI, LLM integration, and vector storage:

```bash
pip install streamlit python-dotenv langchain-groq langchain-huggingface langchain-community faiss-cpu pypdf

```

### 3. Setup Directory

Create a folder named `research_papers` in the same directory and add your PDF files there.

```bash
mkdir research_papers
# (Add your .pdf files into this folder)

```

---

## 📂 Code Analysis

The code builds a "Chat with PDF" tool. Here is a step-by-step breakdown of how it works.

### **1. Environment & Configuration**

The script starts by loading API keys from a `.env` file.

* **Groq API Key**: Authenticates requests to the LLM.
* **HuggingFace Token**: Authenticates requests to download embedding models (if required for gated models, though `all-MiniLM-L6-v2` is public).

### **2. LLM Initialization**

* **Code**: `ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")`
* **Purpose**: Initializes the Language Model. It uses a specific version of Llama 3 via Groq for ultra-fast response generation.

### **3. Vector Embedding Creation (The "Brain")**

This function (`create_vector_embedding`) prepares the data so the AI can search it.

* **Embeddings**: Uses `HuggingFaceEmbeddings` ("all-MiniLM-L6-v2") to convert text into numbers (vectors).
* **Loader**: `PyPDFDirectoryLoader("research_papers")` scans the specific folder and loads all PDF text.
* **Splitter**: `RecursiveCharacterTextSplitter` chops the large PDF text into smaller chunks (1000 characters each). This is necessary because LLMs cannot read entire books in one go; they need bite-sized pieces.
* **Vector Store (FAISS)**:
* `FAISS.from_documents(...)` takes the text chunks and their vector representations and saves them into a **FAISS** index.
* **Why FAISS?** It is a library developed by Facebook AI for efficient similarity search. It allows the app to quickly find the specific paragraph in a PDF that matches your question.


* **Session State**: The code stores the vectors in `st.session_state.vectors` so they don't disappear when you interact with the app (Streamlit reruns the script on every interaction).

### **4. The RAG Chain**

This function (`rag_chain`) defines the logic for answering questions.

1. **Retrieve**: `vectors.as_retriever()` finds the most relevant text chunks from the FAISS index based on the user's query.
2. **Combine**: It joins the retrieved chunks into a single string (`context`).
3. **Generate**: It sends the `context` and the `input` (question) to the LLM via the `prompt`.

### **5. Streamlit UI**

* **`st.button("Create Document Embeddings")`**: Triggers the data processing. You must click this once after adding PDFs.
* **`st.text_input`**: The search bar for your question.
* **Response Display**:
* It shows the AI's answer using `st.write`.
* **Response Time**: Calculates how long the retrieval and generation took.
* **Expander (`st.expander`)**: A "Show More" section that reveals the exact chunks of text the AI used to answer the question. This helps verify the accuracy of the answer.



---

## 📚 Libraries Used

| Library | Purpose in this App |
| --- | --- |
| **`streamlit`** | Creates the web interface (buttons, input fields, display areas). |
| **`python-dotenv`** | Securely loads the API keys from the `.env` file. |
| **`langchain_groq`** | Connects the application to the Groq API for the LLM. |
| **`langchain_huggingface`** | Provides the `HuggingFaceEmbeddings` model used to turn text into vectors. |
| **`langchain_community`** | Contains integrations for third-party tools: `PyPDFDirectoryLoader` (loading PDFs) and `FAISS` (storing vectors). |
| **`faiss-cpu`** | The underlying engine that performs the fast similarity search on the vectors. |
| **`pypdf`** | A dependency required by `PyPDFDirectoryLoader` to read PDF files. |

---

## 🚀 How to Run

1. Place your PDFs in a folder named `research_papers`.
2. Run the Streamlit app:
```bash
streamlit run app.py

```


3. **In the Browser**:
* Click **"Create Document Embeddings"** (Wait for the success message).
* Type a question in the text box (e.g., "What is the conclusion of the paper?").
* View the answer and the source text chunks.