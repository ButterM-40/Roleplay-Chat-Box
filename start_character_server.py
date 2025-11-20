#!/usr/bin/env python3
"""
Start the Character-Focused Roleplay Chat Server
"""

import os
import sys
import uvicorn
import logging

# Add the backend to Python path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start the server"""
    logger.info("🎭 Starting Character-Focused Roleplay Chat Server...")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Import the FastAPI app
        from backend.main import app
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info"
        )
    except ImportError as e:
        logger.error(f"Failed to import app: {e}")
        logger.error("Make sure you're in the correct directory and have all dependencies installed")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

if __name__ == "__main__":
    main()