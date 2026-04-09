import os
import io
import pandas as pd
from typing import List, Dict, Any, Tuple
from docx import Document
from pptx import Presentation
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document as LangchainDocument

from mcp import MCPMessage

class IngestionAgent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _extract_text(self, file_content: bytes, filename: str) -> str:
        ext = filename.split('.')[-1].lower()
        text = ""
        try:
            if ext == "pdf":
                reader = PdfReader(io.BytesIO(file_content))
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif ext == "docx" or ext == "doc":
                doc = Document(io.BytesIO(file_content))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif ext == "pptx" or ext == "ppt":
                ppt = Presentation(io.BytesIO(file_content))
                for slide in ppt.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
            elif ext == "csv":
                df = pd.read_csv(io.BytesIO(file_content))
                text = df.to_string()
            elif ext in ["txt", "md"]:
                text = file_content.decode("utf-8", errors="ignore")
            else:
                text = f"Unsupported file type: {ext}"
        except Exception as e:
            text = f"Error reading {filename}: {str(e)}"
        
        return text

    def process(self, message: MCPMessage) -> MCPMessage:
        if message.type != "INGEST_REQUEST":
            raise ValueError(f"IngestionAgent does not support message type {message.type}")
        
        file_content = message.payload.get("file_content")
        filename = message.payload.get("filename")
        
        raw_text = self._extract_text(file_content, filename)
        
        docs = [LangchainDocument(page_content=raw_text, metadata={"source": filename})]
        chunks = self.text_splitter.split_documents(docs)
        
        payload_chunks = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
        
        response_msg = MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type="DOCUMENT_CHUNKS",
            trace_id=message.trace_id,
            payload={
                "chunks": payload_chunks
            }
        )
        return response_msg

class RetrievalAgent:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None

    def process(self, message: MCPMessage) -> MCPMessage:
        if message.type == "DOCUMENT_CHUNKS":
            chunks_data = message.payload.get("chunks", [])
            docs = [LangchainDocument(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in chunks_data]
            
            if docs:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(docs, self.embeddings)
                else:
                    self.vector_store.add_documents(docs)
            
            return MCPMessage(
                sender="RetrievalAgent",
                receiver="System",
                type="INGESTION_COMPLETE",
                trace_id=message.trace_id,
                payload={"status": "success", "chunks_added": len(docs)}
            )
            
        elif message.type == "RETRIEVE_REQUEST":
            query = message.payload.get("query")
            top_k = message.payload.get("top_k", 3)
            
            if not self.vector_store:
                return MCPMessage(
                    sender="RetrievalAgent",
                    receiver="LLMResponseAgent",
                    type="CONTEXT_RESPONSE",
                    trace_id=message.trace_id,
                    payload={"query": query, "context_chunks": [], "error": "No documents uploaded."}
                )
                
            results = self.vector_store.similarity_search(query, k=top_k)
            context_chunks = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
            
            return MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                type="CONTEXT_RESPONSE",
                trace_id=message.trace_id,
                payload={"query": query, "context_chunks": context_chunks}
            )
        else:
            raise ValueError(f"RetrievalAgent does not support message type {message.type}")

class LLMResponseAgent:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=model_name,
            groq_api_key=groq_api_key
        )

    def process(self, message: MCPMessage) -> MCPMessage:
        if message.type != "CONTEXT_RESPONSE":
            raise ValueError(f"LLMResponseAgent does not support message type {message.type}")
            
        query = message.payload.get("query")
        context_chunks = message.payload.get("context_chunks", [])
        error = message.payload.get("error")
        
        if error:
            response_text = f"Error: {error}"
        else:
            context_text = "\n\n".join([f"Source: {c['metadata'].get('source', 'Unknown')}\n{c['page_content']}" for c in context_chunks])
            
            prompt = f"Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            
            try:
                response = self.llm.invoke(prompt)
                response_text = response.content
            except Exception as e:
                response_text = f"Error querying LLM: {str(e)}"
        
        return MCPMessage(
            sender="LLMResponseAgent",
            receiver="UI",
            type="FINAL_ANSWER",
            trace_id=message.trace_id,
            payload={
                "answer": response_text,
                "sources": context_chunks
            }
        )
