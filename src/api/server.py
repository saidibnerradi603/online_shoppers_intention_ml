"""
API Server Module

This module provides a function to start the FastAPI server
using Uvicorn.
"""
import uvicorn
import sys
import os
from pathlib import Path
from typing import Optional

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent  
sys.path.append(str(parent_dir))

def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server with Uvicorn
    
    Args:
        host: Host address to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload on code changes
    """
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    start_api_server(reload=True)
