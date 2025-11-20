#!/usr/bin/env python3
"""
Individual LoRA Adapter Test Script
Tests each character adapter separately to check for proper character-specific responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import sys
import logging
import json
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

class IndividualAdapterTester:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.base_model_path = "Qwen/Qwen3-0.6B"
        self.adapters_path = "./lora_adapters"
        self.test_prompts = {
            "moses": [
                "Tell me about the Ten Commandments",
                "What happened when you parted the Red Sea?", 
                "How did you receive the law on Mount Sinai?",
                "What would you say to someone who has lost faith?",
                "Describe your experience leading the Israelites through the desert"
            ],
            "samsung_employee": [
                "Tell me about the latest Galaxy phone features",
                "What makes Samsung cameras special?",
                "How does Samsung DeX work?",
                "Compare Samsung with Apple products",
                "What's new in Samsung's foldable technology?"
            ],
            "jinx": [
                "What's your favorite weapon to build?",
                "Tell me about your sister Vi",
                "What happened in Piltover?",
                "Describe your workshop",
                "How do you feel about Silco?"
            ]
        }
        
    def load_base_model(self):
        """Load the base Qwen model without any adapters"""
        logger.info("🔧 Loading base model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Use CPU to avoid GPU memory issues when loading multiple adapters
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                use_cache=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("✅ Base model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load base model: {e}")
            return False
    
    def test_base_model(self, prompt: str) -> str:
        """Test the base model without any adapter"""
        try:
            # Format prompt simply
            formatted_prompt = f"Human: {prompt}\n\n"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean response
            stop_phrases = ["Human:", "\nHuman:", "User:", "\nUser:"]
            for stop_phrase in stop_phrases:
                if stop_phrase in response:
                    response = response.split(stop_phrase)[0].strip()
                    
            return response
            
        except Exception as e:
            logger.error(f"Error generating base model response: {e}")
            return f"Error: {e}"
    
    def test_adapter(self, character_id: str, prompt: str) -> str:
        """Test a specific LoRA adapter"""
        adapter_path = os.path.join(self.adapters_path, character_id)
        
        if not os.path.exists(adapter_path):
            return f"❌ Adapter not found: {adapter_path}"
            
        try:
            logger.info(f"Loading adapter for {character_id}...")
            
            # Load adapter
            model_with_adapter = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=character_id,
                is_trainable=False
            )
            
            # Format prompt
            formatted_prompt = f"Human: {prompt}\n\n"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model_with_adapter.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean response
            stop_phrases = ["Human:", "\nHuman:", "User:", "\nUser:"]
            for stop_phrase in stop_phrases:
                if stop_phrase in response:
                    response = response.split(stop_phrase)[0].strip()
            
            # Clean up the adapter model to free memory
            del model_with_adapter
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return response
            
        except Exception as e:
            logger.error(f"Error testing adapter {character_id}: {e}")
            return f"Error: {e}"
    
    def analyze_adapter_config(self, character_id: str) -> Dict:
        """Analyze adapter configuration"""
        adapter_path = os.path.join(self.adapters_path, character_id)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        
        if not os.path.exists(config_path):
            return {"error": "Config file not found"}
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            return {
                "peft_type": config.get("peft_type"),
                "r": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "lora_dropout": config.get("lora_dropout"),
                "target_modules": config.get("target_modules"),
                "base_model": config.get("base_model_name_or_path"),
                "inference_mode": config.get("inference_mode")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_adapter_file_info(self, character_id: str) -> Dict:
        """Get adapter file information"""
        adapter_path = os.path.join(self.adapters_path, character_id)
        
        if not os.path.exists(adapter_path):
            return {"error": "Adapter directory not found"}
            
        files = {}
        important_files = [
            "adapter_config.json",
            "adapter_model.safetensors", 
            "training_metadata.json"
        ]
        
        for file in important_files:
            file_path = os.path.join(adapter_path, file)
            files[file] = {
                "exists": os.path.exists(file_path),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            
        return files
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all adapters"""
        if not self.load_base_model():
            return
            
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE LORA ADAPTER TEST")
        print("="*80)
        
        # Test each character
        for character_id in ["moses", "samsung_employee", "jinx"]:
            print(f"\n🎭 TESTING CHARACTER: {character_id.upper()}")
            print("-" * 60)
            
            # Adapter info
            config = self.analyze_adapter_config(character_id)
            files = self.get_adapter_file_info(character_id)
            
            print(f"📁 Adapter Info:")
            print(f"   Config: {config}")
            print(f"   Files: {files}")
            
            # Test with character-specific prompts
            print(f"\n💬 Testing with {character_id} prompts:")
            
            for i, prompt in enumerate(self.test_prompts[character_id][:3], 1):
                print(f"\n   Test {i}: {prompt}")
                
                # Test base model
                base_response = self.test_base_model(prompt)
                print(f"   📝 Base Model: {base_response[:100]}...")
                
                # Test adapter
                adapter_response = self.test_adapter(character_id, prompt)
                print(f"   🎭 {character_id.title()}: {adapter_response[:100]}...")
                
                # Quick comparison
                if base_response != adapter_response:
                    print(f"   ✅ Responses differ (adapter working)")
                else:
                    print(f"   ⚠️  Responses identical (potential issue)")
        
        # Cross-character contamination test
        print(f"\n🔄 CROSS-CHARACTER CONTAMINATION TEST")
        print("-" * 60)
        
        test_prompt = "Tell me about yourself"
        
        for character_id in ["moses", "samsung_employee", "jinx"]:
            response = self.test_adapter(character_id, test_prompt)
            print(f"\n{character_id.title()}: {response}")
            
            # Check for other character names in response
            other_chars = [c for c in ["moses", "samsung", "jinx", "vi", "silco"] if c != character_id]
            contamination = []
            
            for other_char in other_chars:
                if other_char.lower() in response.lower():
                    contamination.append(other_char)
                    
            if contamination:
                print(f"   ⚠️  Possible contamination: {contamination}")
            else:
                print(f"   ✅ No obvious contamination detected")
        
        print(f"\n" + "="*80)
        print("🎯 TEST COMPLETE")
        print("="*80)

def main():
    """Main test function"""
    tester = IndividualAdapterTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()