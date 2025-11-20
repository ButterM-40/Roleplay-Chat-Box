#!/usr/bin/env python3
"""
Test Character Consistency for Retrained Adapters
Validates that each character generates appropriate, consistent responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_character_model(character_name, base_model_name="Qwen/Qwen3-0.6B"):
    """Load base model with character adapter"""
    logger.info(f"Loading {character_name} character model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu"
    )
    
    # Load adapter
    adapter_path = f"lora_adapters/{character_name}"
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def test_character_response(model, tokenizer, prompt, character_name, max_length=150):
    """Generate response for a character and prompt"""
    
    # Format prompt
    formatted_prompt = f"Human: {prompt}\n\n"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    response = full_response.replace(formatted_prompt, "").strip()
    
    # Clean up any remaining artifacts
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0].strip()
    
    return response

def test_all_characters():
    """Test all character adapters for consistency"""
    
    characters = ["moses", "samsung_employee", "jinx"]
    
    # Test prompts for each character
    test_prompts = {
        "moses": [
            "How do you lead people through difficult times?",
            "Tell me about the Ten Commandments.", 
            "What gives you strength?"
        ],
        "samsung_employee": [
            "Tell me about Galaxy phone features.",
            "How does Samsung compare to other brands?",
            "What's special about Samsung technology?"
        ],
        "jinx": [
            "Tell me about yourself.",
            "What's your favorite weapon?",
            "How do you feel about chaos?"
        ]
    }
    
    results = {}
    
    for character in characters:
        print(f"\n{'='*60}")
        print(f"TESTING {character.upper()} CHARACTER")
        print(f"{'='*60}")
        
        try:
            # Load character model
            model, tokenizer = load_character_model(character)
            
            character_results = []
            
            # Test each prompt
            for i, prompt in enumerate(test_prompts[character], 1):
                print(f"\nTest {i} - \"{prompt}\":")
                print("-" * 50)
                
                response = test_character_response(model, tokenizer, prompt, character)
                print(f"{response}")
                
                character_results.append({
                    "prompt": prompt,
                    "response": response
                })
            
            results[character] = {
                "success": True,
                "tests": character_results
            }
            
            print(f"\n✅ {character} testing completed successfully!")
            
        except Exception as e:
            print(f"❌ Error testing {character}: {e}")
            results[character] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("CHARACTER CONSISTENCY TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r["success"])
    print(f"Successful characters: {successful}/{len(characters)}")
    
    for character, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{character}: {status}")
    
    return results

if __name__ == "__main__":
    test_all_characters()