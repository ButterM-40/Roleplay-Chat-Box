#!/usr/bin/env python3

import os
import json
import torch
import tempfile
import shutil
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_lora_loading():
    """Debug exactly what's happening with LoRA loading"""
    
    print("🔬 DEBUGGING LORA LOADING PROCESS")
    print("=" * 60)
    
    # Test with Moses adapter
    character_id = "moses"
    base_model_path = "Qwen/Qwen3-0.6B"
    adapter_path = f"lora_adapters/{character_id}"
    
    print(f"📁 Checking adapter files for {character_id}:")
    adapter_model_path = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    
    print(f"  - adapter_model.safetensors: {os.path.exists(adapter_model_path)} ({os.path.getsize(adapter_model_path) if os.path.exists(adapter_model_path) else 0} bytes)")
    print(f"  - adapter_config.json: {os.path.exists(adapter_config_path)} ({os.path.getsize(adapter_config_path) if os.path.exists(adapter_config_path) else 0} bytes)")
    
    if not (os.path.exists(adapter_model_path) and os.path.exists(adapter_config_path)):
        print("❌ Missing adapter files!")
        return
        
    # Load and examine the config
    print(f"\n📋 Adapter configuration:")
    with open(adapter_config_path, 'r') as f:
        config_data = json.load(f)
    
    print(json.dumps(config_data, indent=2))
    
    # Try loading base model
    print(f"\n🤖 Loading base model: {base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu",
            local_files_only=False
        )
        print("✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return
        
    # Clean the config (remove problematic parameters)
    print(f"\n🧹 Cleaning adapter config...")
    problematic_params = [
        'alora_invocation_tokens', 'arrow_config', 
        'ensure_weight_tying', 'peft_version', 'corda_config',
        'eva_config', 'megatron_config', 'megatron_core',
        'loftq_config', 'qalora_group_size'
    ]
    
    cleaned_config = config_data.copy()
    removed_params = []
    for param in problematic_params:
        if param in cleaned_config:
            del cleaned_config[param]
            removed_params.append(param)
    
    if removed_params:
        print(f"  Removed parameters: {removed_params}")
    else:
        print("  No problematic parameters found")
    
    # Create temp directory with cleaned config
    temp_dir = tempfile.mkdtemp()
    temp_config_file = os.path.join(temp_dir, "adapter_config.json")
    temp_model_file = os.path.join(temp_dir, "adapter_model.safetensors")
    
    with open(temp_config_file, 'w') as f:
        json.dump(cleaned_config, f, indent=2)
    
    shutil.copy2(adapter_model_path, temp_model_file)
    
    print(f"  Created temp adapter at: {temp_dir}")
    
    # Try loading the LoRA adapter
    print(f"\n🎭 Attempting to load LoRA adapter...")
    try:
        lora_model = PeftModel.from_pretrained(
            base_model,
            temp_dir,
            adapter_name=f"{character_id}_test",
            is_trainable=False,
            torch_dtype=torch.float32,
        )
        print("✅ LoRA adapter loaded successfully!")
        
        # Test if weights actually changed
        print(f"\n⚖️ Testing if LoRA weights are applied...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Test generation
        test_prompt = "Tell me about the Ten Commandments"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Generate with base model
        print("📝 Base model response:")
        with torch.no_grad():
            base_outputs = base_model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        print(f"   {base_response[len(test_prompt):].strip()}")
        
        # Generate with LoRA model  
        print("🎭 LoRA model response:")
        with torch.no_grad():
            lora_outputs = lora_model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        print(f"   {lora_response[len(test_prompt):].strip()}")
        
        if base_response != lora_response:
            print("✅ Responses differ - LoRA is working!")
        else:
            print("❌ Responses identical - LoRA may not be applied")
            
        # Check model parameters
        print(f"\n🔍 Checking model parameters...")
        base_params = sum(p.numel() for p in base_model.parameters())
        lora_params = sum(p.numel() for p in lora_model.parameters())
        print(f"  Base model parameters: {base_params:,}")
        print(f"  LoRA model parameters: {lora_params:,}")
        print(f"  Parameter difference: {lora_params - base_params:,}")
        
    except Exception as e:
        print(f"❌ Failed to load LoRA adapter: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temp directory")

if __name__ == "__main__":
    debug_lora_loading()