import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from mcp import MCPMessage
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent

load_dotenv()

st.set_page_config(page_title="Agentic RAG Chatbot (MCP)", layout="wide")

@st.cache_resource
def get_agents():
    # Cache agents to maintain state of RetrievalAgent's vector DB
    return {
        "ingestion": IngestionAgent(),
        "retrieval": RetrievalAgent()
    }

def main():
    st.title("🤖 Agentic RAG Chatbot with MCP")
    st.markdown("Upload diverse document formats and query them using an Agentic RAG architecture communicating via Model Context Protocol (MCP).")
    
    agent_instances = get_agents()
    
    with st.sidebar:
        st.header("1. Configuration")
        groq_api_key = st.text_input("Groq API Key (for Llama model):", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        selected_model = st.selectbox(
            "Select Groq Model:",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "llama3-70b-8192",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
            index=0
        )
        
        st.header("2. Upload Documents")
        st.markdown("Supported: PDF, PPTX, CSV, DOCX, TXT/MD")
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Agents are digesting the documents..."):
                    session_chunks = 0
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.read()
                        
                        # 1. Initiate Ingestion -> Retrieval
                        ingest_req = MCPMessage(
                            sender="UI",
                            receiver="IngestionAgent",
                            type="INGEST_REQUEST",
                            payload={
                                "file_content": file_bytes,
                                "filename": uploaded_file.name
                            }
                        )
                        st.session_state.mcp_trace.append(ingest_req.model_dump())
                        
                        chunk_msg = agent_instances["ingestion"].process(ingest_req)
                        st.session_state.mcp_trace.append(chunk_msg.model_dump())
                        
                        # 2. Retrieval store chunks
                        ingest_complete_msg = agent_instances["retrieval"].process(chunk_msg)
                        st.session_state.mcp_trace.append(ingest_complete_msg.model_dump())

                        file_chunks = ingest_complete_msg.payload.get("chunks_added", 0)
                        session_chunks += file_chunks
                        st.session_state.total_files += 1
                    
                    st.session_state.total_chunks += session_chunks
                    st.success(f"✅ Processed **{len(uploaded_files)}** file(s) → **{session_chunks}** chunks added!")

        # Document Stats
        st.header("📊 Document Stats")
        col1, col2 = st.columns(2)
        col1.metric("📄 Files Uploaded", st.session_state.get("total_files", 0))
        col2.metric("🧩 Total Chunks", st.session_state.get("total_chunks", 0))

        st.header("MCP Communication Trace")
        if st.checkbox("Show MCP Logs"):
            if st.session_state.get("mcp_trace"):
                for m in reversed(st.session_state.mcp_trace):
                    st.json(m)
            else:
                st.info("No messages yet.")

    # Main Chat Area
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "mcp_trace" not in st.session_state:
        st.session_state.mcp_trace = []

    if "total_files" not in st.session_state:
        st.session_state.total_files = 0

    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                srcs = message["sources"]
                st.caption(f"🔍 Retrieved **{len(srcs)}** chunk(s) from **{len(set(d['metadata'].get('source','') for d in srcs))}** document(s).")
                with st.expander("🧩 View Retrieved Chunks"):
                    for i, doc in enumerate(srcs):
                        st.markdown(f"**Chunk {i+1} — Source: `{doc['metadata'].get('source', 'Unknown')}`**")
                        st.info(doc['page_content'])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not groq_api_key:
            st.error("Please provide a Groq API Key in the sidebar.")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agents are collaborating to answer..."):
                # Run the Agentic communication
                llm_agent = LLMResponseAgent(groq_api_key=groq_api_key, model_name=selected_model)
                
                # 3. Retrieve request
                retrieve_req = MCPMessage(
                    sender="UI",
                    receiver="RetrievalAgent",
                    type="RETRIEVE_REQUEST",
                    payload={"query": prompt}
                )
                st.session_state.mcp_trace.append(retrieve_req.model_dump())
                
                context_msg = agent_instances["retrieval"].process(retrieve_req)
                st.session_state.mcp_trace.append(context_msg.model_dump())
                
                # 4. LLM Generation
                final_answer_msg = llm_agent.process(context_msg)
                st.session_state.mcp_trace.append(final_answer_msg.model_dump())
                
                answer = final_answer_msg.payload.get("answer", "")
                sources = final_answer_msg.payload.get("sources", [])
                
                st.markdown(answer)

                # Show chunk retrieval stats
                num_chunks = len(sources)
                if num_chunks > 0:
                    st.caption(f"🔍 Retrieved **{num_chunks}** chunk(s) from **{len(set(d['metadata'].get('source','') for d in sources))}** document(s) to generate this answer.")
                    with st.expander("🧩 View Retrieved Chunks"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1} — Source: `{doc['metadata'].get('source', 'Unknown')}`**")
                            st.info(doc['page_content'])
                else:
                    st.caption("⚠️ No chunks retrieved — no documents have been uploaded yet.")

                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

if __name__ == "__main__":
    main()
