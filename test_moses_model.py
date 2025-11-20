#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')

from backend.models.character_manager import CharacterManager
from backend.config import settings
import logging

logging.basicConfig(level=logging.INFO)

async def test_moses_model():
    """Test what model Moses is actually using"""
    
    print("🔍 TESTING MOSES MODEL IN CHARACTER MANAGER")
    print("=" * 60)
    
    # Initialize character manager (same as server)
    manager = CharacterManager()
    await manager.initialize()
    
    print(f"\n📋 Available characters: {manager.get_available_characters()}")
    
    # Check Moses model
    if 'moses' in manager.character_models:
        moses_model = manager.character_models['moses']
        print(f"\n🎭 Moses model type: {type(moses_model)}")
        print(f"   Model class: {moses_model.__class__.__name__}")
        
        # Check if it's a PeftModel (LoRA) or base model
        if hasattr(moses_model, 'base_model'):
            print("✅ Moses has LoRA adapter (PeftModel detected)")
            print(f"   Base model: {type(moses_model.base_model)}")
            if hasattr(moses_model, 'peft_config'):
                print(f"   LoRA config: {moses_model.peft_config}")
        else:
            print("❌ Moses is using base model (no LoRA)")
        
        # Test generation directly
        print(f"\n💬 Testing Moses generation...")
        response = await manager.generate_response(
            character_id='moses',
            user_message='Tell me about the Ten Commandments',
            conversation_history=[]
        )
        print(f"Moses response: {response}")
    else:
        print("❌ Moses not found in character models!")

if __name__ == "__main__":
    asyncio.run(test_moses_model())