import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------- ENV SETUP --------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# -------------------- LLM --------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
)

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question strictly using the provided context.
    Be concise and accurate.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

output_parser = StrOutputParser()

# -------------------- VECTOR CREATION --------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        documents = st.session_state.loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_docs = splitter.split_documents(documents[:50])

        st.session_state.vectors = FAISS.from_documents(
            final_docs,
            st.session_state.embeddings
        )

# -------------------- RAG PIPELINE --------------------
def rag_chain(query: str):
    retriever = st.session_state.vectors.as_retriever()
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | output_parser
    return chain.invoke({"context": context, "input": query}), docs

# -------------------- STREAMLIT UI --------------------
st.title("📄 RAG Document Q&A with Groq + LangChain (Modern)")

user_prompt = st.text_input("Ask a question from the research papers")

if st.button("Create Document Embeddings"):
    create_vector_embedding()
    st.success("Vector database created successfully")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first.")
    else:
        start = time.process_time()
        answer, source_docs = rag_chain(user_prompt)
        elapsed = time.process_time() - start

        st.subheader("Answer")
        st.write(answer)

        st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

        with st.expander("🔍 Document Similarity Search"):
            for i, doc in enumerate(source_docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")