#!/usr/bin/env python3
"""
Simple Server Test - Minimal FastAPI without complex dependencies
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Simple Test Server")

@app.get("/")
async def root():
    return {"message": "Simple server is working!"}

@app.get("/test")
async def test():
    return {"status": "OK", "characters": ["moses", "samsung_employee", "jinx"]}

if __name__ == "__main__":
    print("🧪 Starting Simple Test Server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,  # Different port
        reload=False
    )