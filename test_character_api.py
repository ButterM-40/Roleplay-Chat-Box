#!/usr/bin/env python3
"""
Quick Character API Test - Demonstrates working character adapters
"""

import requests
import time

def test_character_chat():
    """Test characters via API (bypassing WebSocket issues)"""
    base_url = "http://127.0.0.1:8000/api/chat"
    
    tests = [
        {
            "character": "moses",
            "message": "How do you lead people through difficult times?",
            "expected": "biblical wisdom"
        },
        {
            "character": "samsung_employee", 
            "message": "Tell me about Galaxy phones",
            "expected": "enthusiastic tech knowledge"
        },
        {
            "character": "jinx",
            "message": "Hey there!",
            "expected": "chaotic energy"
        }
    ]
    
    print("🎭 Testing Character-Focused Adapters via API")
    print("=" * 60)
    
    for test in tests:
        print(f"\n🎯 Testing {test['character'].upper()}")
        print(f"Question: {test['message']}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/{test['character']}",
                json={
                    "text": test['message'],
                    "timestamp": int(time.time() * 1000)
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                character_response = result["response"]
                print(f"✅ Response: {character_response[:200]}{'...' if len(character_response) > 200 else ''}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    print(f"\n{'=' * 60}")
    print("✅ Character API Testing Complete!")
    print("🔧 Note: WebSocket needs fixing for real-time chat")

if __name__ == "__main__":
    test_character_chat()