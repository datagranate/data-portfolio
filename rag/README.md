![](../images/banner.png)

# FCA regulatory compliance RAG pipeline demo

A Retrieval-Augmented Generation (RAG) application designed to answer questions about the UK Financial Conduct Authority (FCA) Handbook, COBS Chapters 1–10A. This project demonstrates end-to-end Generative AI engineering, from document ingestion to a deployed Streamlit interface.

## Objective
To enable financial professionals to query complex regulatory text using natural language, ensuring accurate, citation-backed responses based on the latest FCA guidance (updated August 2026).


## Architecture and tech stack
*   **Data source:** UK FCA Handbook, chapters 1-10A of the Conduct of Business Sourcebook (COBS) section, see [FCA Handbook](https://handbook.fca.org.uk/handbook)
*   **Ingestion and cleaning:** Custom Python code to remove footers, headers and artifacts from raw PDFs
*   **Splitting:** LangChain `RecursiveCharacterTextSplitter` for semantic chunking
*   **Embeddings:** Hugging Face's `all-MiniLM-L6-v2` for dense vector representation
*   **Vector store:** FAISS (Facebook AI Similarity Search) for efficient similarity search
    *   *Note:* Currently using FAISS via `langchain-community` which is being sunsetted; planned migration, probably to ChromaDB
*   **LLM:** ChatGroq (model: `compound-mini`), configured with `temperature=0` for deterministic, factual responses
*   **Deployment:** Streamlit (`streamlit_app.py`) for an interactive web interface

