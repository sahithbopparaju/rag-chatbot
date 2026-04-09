README — Agentic RAG Chatbot (MCP)

Project Title:
Agentic RAG Chatbot with Model Context Protocol (MCP)

Overview:
This project is an Agent-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF, DOCX, PPTX, CSV, TXT) and ask questions based on their content.

The system uses a multi-agent architecture where different agents handle ingestion, retrieval, and response generation, communicating via a structured protocol called MCP (Model Context Protocol).


🧠 Key Features
•	📂 Upload multiple document formats: 
o	PDF, DOCX, PPTX, CSV, TXT, MD 
•	🔍 Semantic search using FAISS vector database 
•	🤖 LLM-powered answers using Groq (LLaMA / Mixtral / Gemma models) 
•	🧩 Chunk-based document processing 
•	🔄 Agent-to-agent communication using MCP 
•	📊 Real-time document statistics 
•	🧾 View retrieved chunks (explainable AI) 
•	🧠 Context-aware Q&A system


Architecture:
Agent Workflow:
User → Ingestion Agent → Retrieval Agent → LLM Response Agent → UI
Components:
1. Ingestion Agent
- Extracts text from uploaded files
- Splits text into chunks using RecursiveCharacterTextSplitter

2. Retrieval Agent
- Converts chunks into embeddings using HuggingFace
- Stores vectors in FAISS
- Retrieves top-k relevant chunks

3. LLM Response Agent
- Uses Groq LLM (LLaMA / Mixtral)
- Generates final answer based on retrieved context

4. MCP (Model Context Protocol)
- Standard message structure for agent communication
- Ensures modular and scalable design

Tech Stack:
Frontend: Streamlit
Backend: Python
LLM: Groq (LLaMA, Mixtral, Gemma)
Embeddings: HuggingFace Transformers
Vector DB: FAISS
Libraries:
LangChain, PyPDF2, python-docx, python-pptx, pandas

Project Structure:
rag/
│── app.py              # Streamlit UI
│── agents.py           # All agents (Ingestion, Retrieval, LLM)
│── mcp.py              # MCP message structure
│── requirements.txt    # Dependencies
│── .env                # API keys (not to be shared)

Installation & Setup:
1. Clone Repository:
git clone <your-repo-url>
cd rag

2. Create Virtual Environment:
python -m venv venv
venv\Scripts\activate

3. Install Dependencies:
pip install -r requirements.txt

4. Add API Key:
Create .env file and add:
GROQ_API_KEY=your_api_key_here

Run the Application:
streamlit run app.py

How It Works:
1. Upload documents
2. Click Process Documents
3. Ask questions in chat

System workflow:
- Retrieves relevant chunks
- Sends context to LLM
- Generates answer

Example Use Cases:
- Academic document Q&A
- Resume/document analysis
- CSV data understanding
- Report summarization

MCP Message Flow:
1. INGEST_REQUEST → UI to IngestionAgent
2. DOCUMENT_CHUNKS → IngestionAgent to RetrievalAgent
3. INGESTION_COMPLETE → RetrievalAgent to System
4. RETRIEVE_REQUEST → UI to RetrievalAgent
5. CONTEXT_RESPONSE → RetrievalAgent to LLMResponseAgent
6. FINAL_ANSWER → LLMResponseAgent to UI

Environment Variables:
GROQ_API_KEY — API key for Groq LLM

Known Limitations:
- No persistent vector DB
- Depends on external LLM
- Limited error handling

Future Improvements:
- Add OpenAI / local LLM support
- Store embeddings in database
- Add authentication
- Improve UI/UX
- Add chat history persistence

Conclusion:
This project demonstrates agent-based system design, RAG implementation, LLM integration, and modular scalable architecture using MCP.

