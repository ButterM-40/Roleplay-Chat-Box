#!/usr/bin/env python3
"""
REST API-Only Character Server
Simple, reliable server using only REST endpoints
"""

import os
import sys
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import traceback

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="Character REST API", version="2.0.0")

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
voice_synthesizer = None
characters_loaded = False
voice_enabled = False

@app.on_event("startup")
async def startup_event():
    """Initialize character manager and voice synthesizer on startup"""
    global character_manager, voice_synthesizer, characters_loaded, voice_enabled
    logger.info("🎭 Initializing REST API Character Server...")
    
    try:
        from backend.models.character_manager import CharacterManager
        
        character_manager = CharacterManager()
        await character_manager.initialize()
        characters_loaded = True
        logger.info("✅ All character adapters loaded successfully!")
        
        logger.info("ℹ️ Voice synthesis disabled - focusing on character chat only")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize characters: {e}")
        logger.error(traceback.format_exc())
        characters_loaded = False

@app.get("/")
async def root():
    """Serve the REST API frontend"""
    frontend_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "rest_index.html")
    
    if os.path.exists(frontend_html_path):
        return FileResponse(frontend_html_path)
    else:
        # Fallback to regular index with modified script
        return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"))

@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "mode": "REST API",
        "characters_loaded": characters_loaded,
        "voice_enabled": voice_enabled,
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
                "description": "Tech expert",
                "avatar": "/static/avatars/samsung.svg"
            },
            {
                "id": "jinx",
                "name": "Jinx",
                "description": "Chaotic genius",
                "avatar": "/static/avatars/jinx.svg"
            }
        ]
    }

@app.post("/api/chat/{character_id}")
async def chat_with_character(character_id: str, message: dict):
    """Send message to specific character via REST API"""
    if not characters_loaded or not character_manager:
        raise HTTPException(
            status_code=503, 
            detail="Character adapters not loaded. Please wait and try again."
        )
        
    if character_id not in ["moses", "samsung_employee", "jinx"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid character ID: {character_id}"
        )
    
    try:
        user_message = message.get("text", "")
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message text is required")
            
        logger.info(f"Generating response for {character_id}: {user_message[:50]}...")
        
        # Generate character response
        response = character_manager.generate_response(
            character_id, 
            user_message,
            message.get("conversation_history", [])
        )
        
        result = {
            "character_id": character_id,
            "response": response,
            "timestamp": message.get("timestamp", 0)
        }
        
        # Generate voice if requested and available
        if message.get("include_voice", False) and voice_enabled and voice_synthesizer:
            try:
                logger.info(f"Generating voice for {character_id}...")
                voice_data = await voice_synthesizer.synthesize(response, character_id)
                if voice_data:
                    result["voice_data"] = voice_data
                    logger.info(f"Voice generated for {character_id}")
                else:
                    logger.warning(f"Voice generation returned None for {character_id}")
            except Exception as ve:
                logger.error(f"Voice generation error for {character_id}: {ve}")
                # Don't fail the whole request if voice fails
        
        logger.info(f"Response generated for {character_id} ({len(response)} chars)")
        return result
        
    except Exception as e:
        logger.error(f"Error generating response for {character_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Character response generation failed: {str(e)}"
        )

@app.get("/api/voice/status")
async def get_voice_status():
    """Get voice synthesis status"""
    return {
        "voice_enabled": voice_enabled,
        "voice_model_loaded": voice_synthesizer is not None,
        "supported_characters": ["moses", "samsung_employee", "jinx"] if voice_enabled else []
    }

@app.get("/api/voice/character/{character_id}")
async def get_character_voice_info(character_id: str):
    """Get voice configuration for a character"""
    if not voice_enabled or not voice_synthesizer:
        raise HTTPException(status_code=503, detail="Voice synthesis not available")
        
    if character_id not in ["moses", "samsung_employee", "jinx"]:
        raise HTTPException(status_code=400, detail=f"Invalid character ID: {character_id}")
    
    voice_info = voice_synthesizer.get_character_voice_info(character_id)
    return {
        "character_id": character_id,
        "voice_config": voice_info,
        "voice_available": voice_enabled
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "characters": characters_loaded,
        "voice": voice_enabled,
        "timestamp": __import__("time").time()
    }

# Mount static files
try:
    frontend_static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "static")
    if os.path.exists(frontend_static_path):
        app.mount("/static", StaticFiles(directory=frontend_static_path), name="static")
        logger.info(f"✅ Static files mounted: {frontend_static_path}")
except Exception as e:
    logger.warning(f"⚠️ Static files not mounted: {e}")

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        raise HTTPException(status_code=404, detail="Favicon not found")

if __name__ == "__main__":
    logger.info("🚀 Starting REST API Character Server...")
    
    # Change to correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8003,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        logger.error(traceback.format_exc())