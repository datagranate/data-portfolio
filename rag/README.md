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
*   **LLM:** ChatGroq (current model: `openai/gpt-oss-20b`), configured with `temperature=0` as a baseline for consistent responses (experimentation planned).
*   **Deployment:** Streamlit (`streamlit_app.py`) for an interactive web interface. 
*   **Evaluation suite:** Built with DeepEval using Groq models. Generated synthetic "golden" test cases calling `groq/compound-mini` (now deprecated), which I am checking against the source material to ensure accuracy. Initial set of approved goldens evaluated using Answer Relevance, Faithfulness and a customer Tone/Citation metric. Uses `qwen/qwen3.8-27b` as the judge model and `openai/gpt-oss-20b` as the chat model; deliberately choosing different models to avoid self-preference bias. Baseline logged with MLFlow, further experimentation planned.

