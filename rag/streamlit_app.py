import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# configuration
INDEX_PATH = "rag/faiss_index" 
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GITHUB_REPO_URL = "https://github.com/datagranate/data-portfolio" 

st.set_page_config(page_title="FCA COBS Compliance Assistant", page_icon="🏛️", layout="wide")
st.title("FCA COBS Compliance Assistant")
st.markdown("""
This app answers questions based on the **UK FCA Handbook (COBS Chapters 1-10A)**.
""")

# Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.warning("**API key missing**: Please set `GROQ_API_KEY` in Streamlit Cloud settings.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

@st.cache_resource
def load_vector_store():
    with st.spinner("Loading compliance knowledge base..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            vector_store = FAISS.load_local(
                INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # quick sanity check (silent)
            if len(vector_store.docstore._dict) == 0:
                st.error("Index is empty. Please check the data source.")
                return None
                
            return vector_store
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            return None

vector_store = load_vector_store()

if vector_store is None:
    st.stop()

llm = ChatGroq(model_name=GROQ_MODEL, temperature=0)

# RAG chain
prompt_template = """You are a professional UK Financial Compliance Assistant. 
Answer the user's question using ONLY the provided context from the FCA COBS handbook (Chapters 1-10A).
    
CRITICAL INSTRUCTIONS:
1. Do NOT start your answer with phrases like "Based on the context," "Based on the provided text," or "According to the documents."
2. Start your answer **directly** with the factual information.
3. If the answer is not in the context, simply state: "I cannot find this information in the provided COBS chapters."
4. Do not hallucinate. Be precise and professional.

Context: {context}
Question: {question}

Helpful answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vector_store.as_retriever(search_kwargs={"k": 6})
retrieval_chain = retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt 
#     | llm 
#     | StrOutputParser()
# )

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# multi-turn chat handler
def build_conversation_history(messages, max_turns=4):
    if not messages:
        return ""
    
    # include the *current* user message in the history string so the model sees the full context of the *current* turn's intent
    recent_messages = messages[-max_turns:]
    
    history_parts = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get('content', '')
        history_parts.append(f"{role}: {content}")
    
    return "\n".join(history_parts)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a RAG system.

Given the chat history and the current question, output a standalone search query.

RULES:
1. If the current question depends on the conversation (e.g., "more details", "what about that rule?", "and exclusions?"), rewrite it to include the relevant context from history.
2. If the current question introduces a NEW TOPIC and can be understood without history, IGNORE the history entirely and output only the current question.
3. Only output the rewritten query, nothing else.""")
    ,
    ("human", """Chat history:
{chat_history}

Current question:
{question}

Standalone search query:""")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

if user_prompt := st.chat_input("Ask a question about COBS (eg, 'What are the rules on inducements?'):"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FCA Handbook..."):
            try:
                chat_history = build_conversation_history(st.session_state.messages)
                
                # rewrite query using conversation history (history exists)
                if chat_history.strip():
                    with st.spinner("Understanding context..."):
                        rewritten_query = rewrite_chain.invoke({
                            "chat_history": chat_history,
                            "question": user_prompt
                        })
                        search_query = rewritten_query.strip()
                else:
                    search_query = user_prompt
                
                # retrieve using rewritten query
                context_docs = retriever.invoke(search_query)
                context_text = "\n\n".join([doc.page_content for doc in context_docs])
               
                response = prompt.format(
                    context=context_text,
                    question=search_query  
                )
                
                # --- DEBUG BLOCK START ---
                if st.session_state.get('debug_mode', False):
                    st.markdown("DEBUG: Search query")
                    st.code(search_query)
                    
                    st.markdown("DEBUG: Retrieved context")
                    st.write(f"Found {len(context_docs)} docs.")
                    if len(context_docs) == 0:
                        st.error("NO DOCS RETRIEVED!")
                    else:
                        for i, doc in enumerate(context_docs):
                            st.text_area(f"Doc {i+1} ({len(doc.page_content)} chars)", value=doc.page_content[:500], key=f"dbg_doc_{i}")
                            
                    st.markdown("DEBUG: Final prompt sent to LLM")
                    st.text_area("Full Prompt", value=response, height=300)
                # --- DEBUG BLOCK END ---
                
                
                response = llm.invoke(response)
                response_text = StrOutputParser().invoke(response)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar
st.sidebar.image("images/datagranate_logo.png")
st.sidebar.success(f"[Visit my data portfolio]({GITHUB_REPO_URL})")

# --- DEBUG TOGGLE START ---
with st.sidebar.expander("Debug Mode"):
    show_debug = st.checkbox("Show Prompt Debug", value=False)
    st.session_state['debug_mode'] = show_debug
# --- DEBUG TOGGLE END ---

st.sidebar.markdown("---")

# about
st.sidebar.header("About this demo")
st.sidebar.markdown("""
This application is a practical demonstration of **RAG pipeline implementation**, focusing on 
**data engineering** and **AI integration**.

**Knowledge base:**
- **Source:** UK Financial Conduct Authority (FCA) Handbook [view FCA Handbook online]("https://handbook.fca.org.uk/handbook")
- **Section:** Conduct of Business Sourcebook (COBS), Chapters 1–10A.
- **Version:** Last updated 5 August 2026.
""".format()
)

st.sidebar.markdown("---")

# disclaimer
st.sidebar.warning("""
**Disclaimer:**
This tool is for **demonstration and portfolio purposes only**. 
- It is **NOT** a legal advice tool.
- Answers are generated by an AI and may contain errors.
- Do not rely on these responses for actual financial compliance decisions.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with LangChain, FAISS, and Streamlit.*")

