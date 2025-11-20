#!/usr/bin/env python3
"""
Test to verify LoRA weights are actually being applied to models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def test_lora_weight_loading():
    """Test if LoRA weights are actually being applied"""
    print("🔬 Testing LoRA Weight Loading...")
    
    base_model_path = "Qwen/Qwen3-0.6B"
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get base model weights for comparison
    base_weights = {}
    for name, param in base_model.named_parameters():
        if 'q_proj' in name or 'v_proj' in name:  # Target modules for LoRA
            base_weights[name] = param.data.clone()
    
    print(f"Found {len(base_weights)} target parameters in base model")
    
    # Test Moses adapter
    moses_adapter_path = "./lora_adapters/moses"
    if os.path.exists(moses_adapter_path):
        print(f"\n🎭 Testing Moses adapter...")
        
        try:
            # Load Moses adapter
            moses_model = PeftModel.from_pretrained(
                base_model,
                moses_adapter_path,
                adapter_name="moses_test",
                is_trainable=False
            )
            
            # Check if weights changed
            changes_found = 0
            for name, param in moses_model.named_parameters():
                if name in base_weights:
                    if not torch.equal(base_weights[name], param.data):
                        changes_found += 1
                        print(f"  ✅ Weight changed: {name}")
                    
            if changes_found > 0:
                print(f"  ✅ Moses adapter applied! {changes_found} parameters modified")
            else:
                print(f"  ❌ Moses adapter NOT applied - no weight changes detected")
                
            # Test generation difference
            test_prompt = "Human: Who are you?\n\n"
            
            # Base model response
            inputs = tokenizer(test_prompt, return_tensors="pt")
            with torch.no_grad():
                base_outputs = base_model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=True)
            base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            
            # Moses model response  
            with torch.no_grad():
                moses_outputs = moses_model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=True)
            moses_response = tokenizer.decode(moses_outputs[0], skip_special_tokens=True)
            
            print(f"\n📝 Base model: {base_response[len(test_prompt):].strip()}")
            print(f"🎭 Moses model: {moses_response[len(test_prompt):].strip()}")
            
            if base_response != moses_response:
                print("✅ Responses differ - adapter is working!")
            else:
                print("❌ Responses identical - adapter may not be applied")
                
        except Exception as e:
            print(f"❌ Error loading Moses adapter: {e}")
    else:
        print("❌ Moses adapter not found")
    
    # Check adapter configuration
    print(f"\n🔧 Checking adapter configurations...")
    for character in ["moses", "samsung_employee", "jinx"]:
        config_path = f"./lora_adapters/{character}/adapter_config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  {character}: r={config.get('r')}, alpha={config.get('lora_alpha')}, targets={config.get('target_modules')}")

if __name__ == "__main__":
    test_lora_weight_loading()