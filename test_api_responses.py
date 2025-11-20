#!/usr/bin/env python3

import requests
import json

def test_character_api():
    """Test the character API to see if LoRA adapters are working"""
    
    base_url = "http://127.0.0.1:8003"
    
    # Test Moses
    print("🎭 Testing Moses character...")
    moses_response = requests.post(f"{base_url}/chat", json={
        "message": "Tell me about the Ten Commandments",
        "character_id": "moses"
    })
    
    if moses_response.status_code == 200:
        result = moses_response.json()
        print(f"✅ Moses response: {result['response']}")
    else:
        print(f"❌ Moses failed: {moses_response.status_code} - {moses_response.text}")
    
    # Test Samsung Employee  
    print(f"\n🎭 Testing Samsung Employee character...")
    samsung_response = requests.post(f"{base_url}/chat", json={
        "message": "Tell me about Samsung Galaxy features",
        "character_id": "samsung_employee"
    })
    
    if samsung_response.status_code == 200:
        result = samsung_response.json()
        print(f"✅ Samsung Employee response: {result['response']}")
    else:
        print(f"❌ Samsung Employee failed: {samsung_response.status_code} - {samsung_response.text}")
        
    # Test Jinx
    print(f"\n🎭 Testing Jinx character...")
    jinx_response = requests.post(f"{base_url}/chat", json={
        "message": "What's your favorite weapon to build?",
        "character_id": "jinx"
    })
    
    if jinx_response.status_code == 200:
        result = jinx_response.json()
        print(f"✅ Jinx response: {result['response']}")
    else:
        print(f"❌ Jinx failed: {jinx_response.status_code} - {jinx_response.text}")

if __name__ == "__main__":
    test_character_api()