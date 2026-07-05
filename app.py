import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from mcp import MCPMessage
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
import time

load_dotenv()

# ============= FAST PAGE CONFIG =============
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= INLINE CSS (Fast) =============
st.markdown("""
<style>
:root {
    --primary: #7C3AED;
    --secondary: #EC4899;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #F9FAFB;
}

.header {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
    color: white;
    padding: 2rem;
    border-radius: 0;
    margin: 0 -1rem 2rem -1rem;
}

.header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.95;
}

.chat-user {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
    color: white;
    padding: 1rem;
    border-radius: 12px 4px 12px 12px;
    margin: 1rem 10% 1rem 0;
}

.chat-assistant {
    background: white;
    color: #1F2937;
    padding: 1rem;
    border-radius: 4px 12px 12px 12px;
    border: 1px solid #E5E7EB;
    margin: 1rem 0 1rem 10%;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.badge-success {
    background: #ECFDF5;
    color: #065F46;
}

.source-card {
    background: white;
    border-left: 4px solid #7C3AED;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

@media (max-width: 768px) {
    .chat-user, .chat-assistant {
        margin-left: 0;
        margin-right: 0;
    }
}
</style>
""", unsafe_allow_html=True)

# ============= ULTRA-FAST CACHING =============
@st.cache_resource(show_spinner=False)
def init_agents():
    """Initialize agents once and reuse"""
    return {
        "ingestion": IngestionAgent(),
        "retrieval": RetrievalAgent()
    }

@st.cache_resource(show_spinner=False)
def get_llm_agent(api_key: str, model: str):
    """Cache LLM agent per config"""
    if not api_key:
        return None
    try:
        return LLMResponseAgent(groq_api_key=api_key, model_name=model)
    except:
        return None

# ============= SESSION STATE =============
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_files" not in st.session_state:
        st.session_state.total_files = 0
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "mcp_trace" not in st.session_state:
        st.session_state.mcp_trace = []

init_state()

# ============= MAIN APP =============
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤖 RAG Chatbot</h1>
        <p>Smart Document Search & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    agents = init_agents()
    
    # ============= SIDEBAR =============
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Get from https://console.groq.com"
        )
        
        selected_model = st.selectbox(
            "🧠 Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
            index=0
        )
        
        st.divider()
        
        st.markdown("## 📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "doc"]
        )
        
        if st.button("🚀 Process", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.error("Upload at least one file")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                chunks_total = 0
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        status.text(f"Processing: {file.name}")
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        file_bytes = file.read()
                        
                        ingest_req = MCPMessage(
                            sender="UI",
                            receiver="IngestionAgent",
                            type="INGEST_REQUEST",
                            payload={"file_content": file_bytes, "filename": file.name}
                        )
                        
                        chunk_msg = agents["ingestion"].process(ingest_req)
                        ingest_complete = agents["retrieval"].process(chunk_msg)
                        
                        chunks = ingest_complete.payload.get("chunks_added", 0)
                        chunks_total += chunks
                        st.session_state.total_files += 1
                    except Exception as e:
                        st.error(f"Error: {str(e)[:50]}")
                
                progress_bar.empty()
                status.empty()
                st.session_state.total_chunks += chunks_total
                st.success(f"✅ {len(uploaded_files)} files → {chunks_total} chunks")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        col1.metric("📄 Files", st.session_state.total_files)
        col2.metric("🧩 Chunks", st.session_state.total_chunks)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ============= CHAT AREA =============
    st.divider()
    
    # Display messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <strong>👤 You</strong><br/>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            status_badge = f"""
            <span class="badge badge-{'success' if msg.get('status') == 'success' else 'error'}">
                {'✅ SUCCESS' if msg.get('status') == 'success' else '❌ ERROR'}
            </span>
            """
            st.markdown(f"""
            <div class="chat-assistant">
                <strong>🤖 Assistant</strong>{status_badge}<br/><br/>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} used)"):
                    for i, src in enumerate(msg["sources"], 1):
                        source_name = src['metadata'].get('source', 'Unknown')
                        preview = src['page_content'][:100].replace('\n', ' ') + "..."
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 {source_name}</strong><br/>
                            <small>{preview}</small>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        if not groq_api_key:
            st.error("Provide Groq API Key in sidebar")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
        <div class="chat-user">
            <strong>👤 You</strong><br/>
            {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Process
        with st.spinner("⏳ Thinking..."):
            try:
                llm_agent = get_llm_agent(groq_api_key, selected_model)
                if not llm_agent:
                    raise Exception("Failed to initialize LLM")
                
                # Retrieve
                retrieve_req = MCPMessage(
                    sender="UI",
                    receiver="RetrievalAgent",
                    type="RETRIEVE_REQUEST",
                    payload={"query": prompt, "top_k": 3}
                )
                context_msg = agents["retrieval"].process(retrieve_req)
                
                # Generate
                final_msg = llm_agent.process(context_msg)
                
                answer = final_msg.payload.get("answer", "No answer")
                sources = final_msg.payload.get("sources", [])
                status = final_msg.payload.get("status", "error")
                
                # Display answer
                status_badge = f"""
                <span class="badge badge-{'success' if status == 'success' else 'error'}">
                    {'✅ SUCCESS' if status == 'success' else '❌ ERROR'}
                </span>
                """
                st.markdown(f"""
                <div class="chat-assistant">
                    <strong>🤖 Assistant</strong>{status_badge}<br/><br/>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                if sources and status == "success":
                    with st.expander(f"📚 Sources ({len(sources)} used)"):
                        for i, src in enumerate(sources, 1):
                            source_name = src['metadata'].get('source', 'Unknown')
                            preview = src['page_content'][:100].replace('\n', ' ') + "..."
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 {source_name}</strong><br/>
                                <small>{preview}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "status": status
                })
            
            except Exception as e:
                error_msg = f"❌ {str(e)[:100]}"
                st.markdown(f"""
                <div class="chat-assistant">
                    <strong>🤖 Assistant</strong>
                    <span class="badge" style="background: #FEF2F2; color: #7F1D1D;">ERROR</span><br/><br/>
                    {error_msg}
                </div>
                """, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "status": "error"
                })

if __name__ == "__main__":
    main()
