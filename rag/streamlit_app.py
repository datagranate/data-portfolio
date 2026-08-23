import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
INDEX_PATH = "chroma_index" 
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "groq/compound-mini"

st.set_page_config(page_title="FCA COBS Compliance Assistant", page_icon="🏛️", layout="wide")
st.title("FCA COBS Compliance Assistant")
st.markdown("""
This app answers questions based on the **UK FCA Handbook (COBS Chapters 1-10A)**.
*Index built via `fca_cobs_indexing_notebook.ipynb`.*
""")

# Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.warning("**API Key Missing**: Please set `GROQ_API_KEY` in Streamlit Cloud settings.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Load pre-built vector store
@st.cache_resource
def load_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        # This looks for the 'faiss_index' folder 
        return Chroma(persist_directory=INDEX_PATH, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None

vector_store = load_vector_store()

if vector_store is None:
    st.stop()

llm = ChatGroq(model_name=GROQ_MODEL, temperature=0)

# RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional UK Financial Compliance Assistant. 
Answer the user's question using ONLY the provided context from the FCA Handbook COBS Chapters 1-10A).
    
CRITICAL INSTRUCTIONS:
1. Do NOT start your answer with phrases like "Based on the context," "Based on the provided text," or "According to the documents."
2. Start your answer **directly** with the factual information.
3. If the answer is not in the context, simply state: "I cannot find this information in the provided COBS chapters."
4. Do not hallucinate. Be precise and professional.

    Context: {context}
    Question: {question}

    Helpful answer:"""),
    ("human", "{question}")
])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about COBS (eg, 'What are the rules on inducements?'):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FCA Handbook..."):
            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar
with st.sidebar:
    st.header("About this app")
    st.write("Data source: **FCA COBS Chapters 1-10A**")
    st.write("Index source: `fca_cobs_indexing_notebook.ipynb`")
    st.write("LLM: **groq/compound-mini** (via Groq)")
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
