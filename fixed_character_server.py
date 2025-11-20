#!/usr/bin/env python3
"""
Fixed Character Server - Handles errors gracefully
"""

import os
import sys
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="Roleplay Chat API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
character_manager = None
characters_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global character_manager, characters_loaded
    logger.info("🎭 Starting Character-Focused Server...")
    
    try:
        # Import here to avoid issues
        from models.character_manager import CharacterManager
        
        character_manager = CharacterManager()
        await character_manager.initialize()
        characters_loaded = True
        logger.info("✅ All character adapters loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize character manager: {e}")
        characters_loaded = False

@app.get("/")
async def root():
    """Serve the frontend HTML file"""
    frontend_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html")
    if os.path.exists(frontend_html_path):
        return FileResponse(frontend_html_path)
    else:
        return {"message": "Roleplay Chat API is running", "characters_loaded": characters_loaded}

@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "characters_loaded": characters_loaded,
        "available_characters": ["moses", "samsung_employee", "jinx"] if characters_loaded else []
    }

@app.get("/api/characters")
async def get_characters():
    """Get list of available characters"""
    return {
        "characters": [
            {
                "id": "moses",
                "name": "Moses",
                "description": "Biblical prophet and lawgiver",
                "avatar": "/static/avatars/moses.svg"
            },
            {
                "id": "samsung_employee", 
                "name": "Samsung Employee",
                "description": "Tech-savvy corporate representative",
                "avatar": "/static/avatars/samsung.svg"
            },
            {
                "id": "jinx",
                "name": "Jinx",
                "description": "Chaotic genius from Arcane",
                "avatar": "/static/avatars/jinx.svg"
            }
        ]
    }

@app.post("/api/chat/{character_id}")
async def chat_with_character(character_id: str, message: dict):
    """Send message to specific character"""
    if not characters_loaded or not character_manager:
        raise HTTPException(status_code=503, detail="Characters not loaded")
        
    try:
        response = character_manager.generate_response(
            character_id, 
            message["text"],
            message.get("conversation_history", [])
        )
        
        result = {
            "character_id": character_id,
            "response": response,
            "timestamp": message.get("timestamp")
        }
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for frontend
try:
    frontend_static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "static")
    if os.path.exists(frontend_static_path):
        app.mount("/static", StaticFiles(directory=frontend_static_path), name="static")
        logger.info(f"✅ Static files mounted from: {frontend_static_path}")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        raise HTTPException(status_code=404, detail="Favicon not found")

if __name__ == "__main__":
    logger.info("🚀 Starting Fixed Character Server...")
    
    # Change to correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Simple WebSocket endpoint for chat"""
    await websocket.accept()
    current_character = "moses"  # Default character
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "switch_character":
                current_character = message["character_id"]
                await websocket.send_text(json.dumps({
                    "type": "character_switched",
                    "character_id": current_character
                }))
                
            elif message["type"] == "chat_message":
                if characters_loaded and character_manager:
                    # Generate response
                    response = character_manager.generate_response(
                        current_character,
                        message["text"],
                        message.get("history", [])
                    )
                    
                    result = {
                        "type": "chat_response",
                        "character_id": current_character,
                        "response": response,
                        "timestamp": message.get("timestamp")
                    }
                    
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Characters not loaded"
                    }))
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")