from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid

class MCPMessage(BaseModel):
    """
    Model Context Protocol (MCP) Message
    Agentic communication structure connecting Ingestion, Retrieval, and LLM Response.
    """
    sender: str
    receiver: str
    type: str # e.g., "PARSE_REQUEST", "CONTEXT_RESPONSE", "FINAL_ANSWER_REQUEST", "FINAL_ANSWER"
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any]
