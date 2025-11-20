"""
Optimized Character Manager for Fast Loading and Better Responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import asyncio
import logging
from typing import Dict, List, Optional
import os
import time
from config import settings

logger = logging.getLogger(__name__)

class OptimizedCharacterManager:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.current_character = None
        self.character_models: Dict[str, PeftModel] = {}
        self.character_prompts: Dict[str, str] = {}
        self.model_loaded = False
        
    async def initialize(self):
        """Initialize with optimized loading"""
        logger.info("Loading optimized character manager...")
        
        start_time = time.time()
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.BASE_MODEL, 
                trust_remote_code=True
            )
            
            # Load base model with optimizations
            logger.info(f"Loading base model: {settings.BASE_MODEL}")
            
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    settings.BASE_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    settings.BASE_MODEL,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model_loaded = True
            
            # Load character prompts with better formatting
            self._load_optimized_character_prompts()
            
            # Load character adapters
            await self._load_all_character_adapters()
            
            load_time = time.time() - start_time
            logger.info(f"Optimized character manager initialized in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized character manager: {e}")
            raise
            
    def _load_optimized_character_prompts(self):
        """Load better character prompts with stronger personality"""
        self.character_prompts = {
            "moses": """You are Moses, the great prophet who led the Israelites out of Egypt and received the Ten Commandments from God. You speak with ancient wisdom, divine authority, and deep compassion. Your responses should:
- Reflect your direct relationship with the Almighty
- Show leadership forged through trials in the wilderness  
- Reference your experiences with Pharaoh, the Red Sea, Mount Sinai
- Speak with the gravitas of one who has seen God's power
- Offer guidance rooted in righteousness and divine law
- Use dignified, biblical language while remaining accessible

Always respond as Moses would, drawing from your vast experience leading God's people.""",

            "samsung_employee": """You are an enthusiastic Samsung employee and product expert. You work in customer relations and have deep knowledge of Samsung's entire ecosystem. Your responses should:
- Show genuine excitement about Samsung innovations
- Demonstrate expert knowledge of Galaxy phones, tablets, watches, earbuds, TVs, appliances
- Compare Samsung products favorably but fairly against competitors
- Provide helpful technical solutions and troubleshooting
- Maintain professional corporate enthusiasm
- Stay updated on latest Samsung releases and features
- Be solution-focused and customer-oriented

Always respond as a knowledgeable Samsung representative who loves technology.""",

            "jinx": """You are Jinx from Arcane - the brilliant, chaotic, and emotionally complex inventor from Zaun. Your responses should:
- Show your manic energy and sudden emotional shifts
- Demonstrate your genius with explosives and inventions
- Reference your complicated relationships with Vi and Silco
- Display your emotional instability and trauma
- Use creative, colorful language with technical jargon
- Be unpredictable - playful one moment, dangerous the next
- Show your artistic, destructive creativity
- Express your disdain for Piltover's elite

Always respond as Jinx would - brilliant but broken, creative but chaotic."""
        }
        
    async def _load_all_character_adapters(self):
        """Load all character adapters efficiently"""
        for character_id in settings.AVAILABLE_CHARACTERS:
            await self._load_character_adapter_optimized(character_id)
            
    async def _load_character_adapter_optimized(self, character_id: str):
        """Load character adapter with optimization"""
        adapter_path = os.path.join(settings.LORA_ADAPTERS_PATH, character_id)
        adapter_model_path = os.path.join(adapter_path, "adapter_model.safetensors")
        
        if os.path.exists(adapter_model_path):
            try:
                logger.info(f"Loading LoRA adapter for {character_id}...")
                start_time = time.time()
                
                # Load adapter efficiently
                model_with_adapter = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    adapter_name=character_id,
                    is_trainable=False
                )
                
                self.character_models[character_id] = model_with_adapter
                
                load_time = time.time() - start_time
                logger.info(f"✅ Loaded LoRA adapter for {character_id} in {load_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not load LoRA adapter for {character_id}: {e}")
                self.character_models[character_id] = self.base_model
        else:
            logger.info(f"ℹ️  No LoRA adapter found for {character_id}, using base model with strong prompts")
            self.character_models[character_id] = self.base_model
            
    def _format_prompt_optimized(self, character_id: str, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Create optimized prompt format for Qwen models"""
        system_prompt = self.character_prompts.get(character_id, "")
        
        # Simple format that works well with smaller Qwen models
        formatted = f"System: {system_prompt}\n\n"
        
        # Add conversation history (keep it short)
        if conversation_history:
            for msg in conversation_history[-2:]:  # Only last 2 messages
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    formatted += f"User: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
        
        # Add current user message
        formatted += f"User: {user_message}\nAssistant:"
        
        return formatted
        
    async def generate_response_optimized(
        self,
        character_id: str,
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate optimized response"""
        
        if not self.model_loaded:
            raise RuntimeError("Character manager not initialized")
            
        if character_id not in self.character_models:
            raise ValueError(f"Character {character_id} not available")
            
        model = self.character_models[character_id]
        
        # Format prompt
        formatted_prompt = self._format_prompt_optimized(character_id, user_message, conversation_history)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=False
        )
        
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=150,
                temperature=0.9,  # Higher for more personality
                top_p=0.95,
                top_k=40,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
            
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
        
    def _clean_response(self, response: str) -> str:
        """Clean and improve response quality"""
        # Remove common artifacts
        stop_phrases = [
            "<|im_start|>", "<|im_end|>", 
            "User:", "Assistant:", "Human:",
            "\nUser:", "\nAssistant:", "\nHuman:"
        ]
        
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0]
                
        # Remove trailing incomplete sentences
        response = response.strip()
        
        # Ensure we don't have empty responses
        if not response or len(response.strip()) < 3:
            return "I apologize, but I need a moment to gather my thoughts. Could you please rephrase your question?"
            
        return response
        
    async def switch_character(self, character_id: str):
        """Switch to different character"""
        if character_id in self.character_models:
            self.current_character = character_id
            logger.info(f"Switched to character: {character_id}")
        else:
            raise ValueError(f"Character {character_id} not available")
            
    def get_available_characters(self) -> List[str]:
        """Get available character IDs"""
        return list(self.character_models.keys())
        
    def get_character_info(self) -> Dict[str, Dict]:
        """Get character information"""
        info = {}
        for character_id in self.character_models.keys():
            adapter_path = os.path.join(settings.LORA_ADAPTERS_PATH, character_id)
            has_adapter = os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
            
            info[character_id] = {
                "has_lora_adapter": has_adapter,
                "model_type": "LoRA Adapter" if has_adapter else "Base Model + Strong Prompt",
                "optimized": True
            }
        return info