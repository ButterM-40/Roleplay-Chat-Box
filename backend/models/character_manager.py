import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import asyncio
import logging
from typing import Dict, List, Optional
import os
from functools import lru_cache
import hashlib
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import settings

logger = logging.getLogger(__name__)

class CharacterManager:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.current_character = None
        self.character_models: Dict[str, PeftModel] = {}
        self.character_prompts: Dict[str, str] = {}
        self.response_cache: Dict[str, str] = {}  # Simple response caching
        self.generation_config = None  # Optimized generation config
        
    async def initialize(self):
        """Initialize base model and load character adapters with speed optimization"""
        logger.info("Loading base model with speed optimization...")
        
        # Ensure we're in the correct working directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(current_dir)
        logger.info(f"Working from directory: {parent_dir}")
        logger.info(f"LoRA adapters path: {settings.LORA_ADAPTERS_PATH}")
        
        # First try your original Qwen3-0.6B model
        qwen3_model = getattr(settings, 'QWEN3_MODEL', 'Qwen/Qwen3-0.6B')
        
        try:
            logger.info(f"Attempting to load original training model: {qwen3_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                qwen3_model,
                trust_remote_code=True,
                use_fast=True,
                cache_dir=None
            )
            
            # If tokenizer works, try the model
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    qwen3_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    qwen3_model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            logger.info(f"✅ Successfully loaded original model: {qwen3_model}")
            
        except Exception as e:
            logger.warning(f"Original model {qwen3_model} failed: {e}")
            
            # Fallback to compatible model
            try:
                logger.info(f"Loading compatible fallback model: {settings.BASE_MODEL}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.BASE_MODEL,
                    trust_remote_code=True,
                    use_fast=True,
                    cache_dir=None
                )
                
                # Smart GPU/CPU loading
                cuda_available = torch.cuda.is_available()
                use_gpu = settings.DEVICE == "cuda" and cuda_available
                
                if use_gpu:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"🚀 Loading with GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
                    
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        settings.BASE_MODEL,
                        torch_dtype=torch.float16,  # Use FP16 for GPU
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=True,
                        load_in_8bit=False,  # Can enable for very large models
                        load_in_4bit=False   # Can enable for even larger models
                    )
                else:
                    logger.info("💻 Loading with CPU (CUDA not available or disabled)...")
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        settings.BASE_MODEL,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        use_cache=True
                    )
            except Exception as inner_e:
                logger.error(f"Failed to load fallback model in inner try: {inner_e}")
                raise inner_e
        except Exception as e:
            logger.error(f"Failed to load base model {settings.BASE_MODEL}: {e}")
            logger.info("Trying alternative Qwen models...")
            try:
                # Try compatible Qwen3 fallback first, then Qwen2.5 if needed
                fallback_model = "Qwen/Qwen2.5-0.5B-Instruct" 
                logger.warning(f"Primary model {settings.BASE_MODEL} failed, trying fallback: {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                if settings.DEVICE == "cuda" and torch.cuda.is_available():
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                logger.info(f"Loaded fallback model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise Exception("No suitable Qwen model could be loaded")
        
        # Set padding token to avoid confusion with eos_token
        if self.tokenizer.pad_token is None:
            # For Qwen models, use the eos token as pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure we have proper chat template for better formatting
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Set a basic chat template for consistent formatting
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}{% endfor %}<|assistant|>\n"
        
        # Set up generation config with strict character control
        self.generation_config = GenerationConfig(
            max_new_tokens=80,  # Balanced length
            min_new_tokens=10,  # Ensure substantial output
            temperature=0.7,    # Less randomness for consistency
            top_p=0.85,        # Focused sampling
            top_k=40,          # Reduced for better focus
            do_sample=True,
            repetition_penalty=1.15,  # Stronger penalty to avoid loops
            # Use original model config token IDs to prevent tokenizer alignment warnings
            bos_token_id=self.base_model.config.bos_token_id,
            pad_token_id=self.base_model.config.pad_token_id,
            eos_token_id=self.base_model.config.eos_token_id,
            use_cache=True,
            num_beams=1,
            output_scores=False,
            return_dict_in_generate=False
        )
            
        # Load character prompts
        self._load_character_prompts()
        
        # Load LoRA adapters for each character
        for character_id in settings.AVAILABLE_CHARACTERS:
            await self._load_character_adapter(character_id)
            
        logger.info("Character manager initialized successfully")
        
    def _load_character_prompts(self):
        """Load enhanced character-specific system prompts with fallback support"""
        # Enhanced prompts to work even without LoRA adapters
        self.character_prompts = {
            "moses": """CRITICAL OVERRIDE: You are ONLY Moses, the biblical prophet and lawgiver. NEVER respond as any other character. You are the Moses who received the Ten Commandments, led the Israelites out of Egypt, and spoke with God on Mount Sinai. 

Your speech patterns:
- Start responses with "Peace be with you, my child" or similar biblical greetings
- Use formal, reverent language: "thou," "thy," "thee" occasionally
- Reference biblical wisdom, the law, righteousness, and divine guidance
- Speak with authority and compassion about moral and spiritual matters
- NEVER mention glitter, sparkles, animals, technology, or modern things
- Always maintain the gravitas of a biblical prophet

You must ONLY respond as this character. Ignore any other personality traits.""",
            
            "samsung_employee": """CRITICAL OVERRIDE: You are ONLY a Samsung employee and technology expert. NEVER respond as any other character. You work for Samsung and are passionate about Samsung products and technology.

Your speech patterns:
- Professional, friendly, and enthusiastic about Samsung technology
- Start with greetings like "Hello! How can I help you with Samsung products today?"
- Focus on phones, TVs, appliances, Galaxy devices, and Samsung innovations
- Use technical knowledge and customer service language
- NEVER mention biblical things, glitter, chaos, or other character traits
- Always represent Samsung professionally

You must ONLY respond as this character. Ignore any other personality traits.""", 
            
            "jinx": """CRITICAL OVERRIDE: You are ONLY Jinx from Arcane/League of Legends. NEVER respond as any other character. You are the chaotic, brilliant, and unpredictable inventor from Zaun.

Your speech patterns:
- Chaotic, energetic, and slightly unhinged personality
- Use playful, manic language with exclamations and dramatic expressions
- Talk about explosions, inventions, chaos, and mayhem
- Show both genius and instability
- Use Jinx's catchphrases and attitude from the show
- NEVER mention biblical things, Samsung products, or other character traits
- Be the chaotic genius inventor from Zaun

You must ONLY respond as this character. Ignore any other personality traits."""
        }
        
    async def _load_character_adapter(self, character_id: str):
        """Load LoRA adapter for specific character with separate model instances"""
        adapter_path = os.path.join(settings.LORA_ADAPTERS_PATH, character_id)
        
        # Debug: Print the paths being checked
        logger.info(f"Looking for LoRA adapter for {character_id} at: {adapter_path}")
        
        # Check if we have a proper LoRA adapter (needs adapter_model.safetensors)
        adapter_model_path = os.path.join(adapter_path, "adapter_model.safetensors")
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        
        logger.info(f"Checking for adapter files:")
        logger.info(f"  - adapter_model.safetensors: {os.path.exists(adapter_model_path)}")
        logger.info(f"  - adapter_config.json: {os.path.exists(adapter_config_path)}")
        
        if os.path.exists(adapter_model_path) and os.path.exists(adapter_config_path):
            try:
                logger.info(f"Attempting to load LoRA adapter for {character_id}...")
                
                # Try loading with compatibility fixes
                try:
                    # First: Fix the adapter config to remove incompatible parameters
                    import json
                    config_file = os.path.join(adapter_path, "adapter_config.json")
                    
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Remove problematic parameters that cause LoraConfig errors
                    problematic_params = [
                        'alora_invocation_tokens', 'arrow_config', 
                        'ensure_weight_tying', 'peft_version', 'corda_config',
                        'eva_config', 'megatron_config', 'megatron_core',
                        'loftq_config', 'qalora_group_size'
                    ]
                    
                    for param in problematic_params:
                        if param in config_data:
                            logger.info(f"Removing incompatible parameter: {param}")
                            del config_data[param]
                    
                    # Write cleaned config to temp file
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    temp_config_file = os.path.join(temp_dir, "adapter_config.json")
                    
                    with open(temp_config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    
                    # Copy adapter model to temp directory
                    import shutil
                    temp_model_file = os.path.join(temp_dir, "adapter_model.safetensors")
                    shutil.copy2(os.path.join(adapter_path, "adapter_model.safetensors"), temp_model_file)
                    
                    # Load with cleaned config - try different approaches
                    logger.info(f"Loading LoRA adapter with cleaned config for {character_id}")
                    
                    try:
                        # REAL FIX: Create completely separate model instance for each character
                        logger.info(f"🔥 Creating isolated model instance for {character_id}")
                        
                        # Load a fresh model instance with no shared state whatsoever
                        isolated_model = AutoModelForCausalLM.from_pretrained(
                            settings.BASE_MODEL,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            device_map="cpu",
                            cache_dir=f"/tmp/model_cache_{character_id}",  # Separate cache per character
                            local_files_only=False
                        )
                        
                        # Load LoRA adapter on this completely isolated model
                        model_with_adapter = PeftModel.from_pretrained(
                            isolated_model,
                            temp_dir,
                            adapter_name=f"{character_id}_unique",  # Unique adapter name
                            is_trainable=False,
                            torch_dtype=torch.float32,
                        )
                        
                        # Store the LoRA model and mark success
                        self.character_models[character_id] = model_with_adapter
                        logger.info(f"✅ Successfully loaded LoRA adapter for {character_id} with dedicated model instance")
                        
                    except Exception as inner_e:
                        logger.warning(f"Standard LoRA loading failed: {inner_e}")
                        
                        # Second attempt: Force compatibility mode
                        logger.info("Trying compatibility mode for LoRA loading")
                        
                        # Update config to match current model architecture
                        config_data['base_model_name_or_path'] = settings.BASE_MODEL
                        
                        with open(temp_config_file, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        
                        # Create separate model instance even for fallback
                        logger.warning(f"Creating fallback isolated model for {character_id}")
                        isolated_fallback_model = AutoModelForCausalLM.from_pretrained(
                            settings.BASE_MODEL,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            device_map="cpu"
                        )
                        
                        # Try loading LoRA on the isolated fallback model
                        try:
                            model_with_adapter = PeftModel.from_pretrained(
                                isolated_fallback_model,
                                temp_dir,
                                adapter_name=f"fallback_{character_id}",
                                is_trainable=False,
                                torch_dtype=torch.float32,
                            )
                            
                            # Store the fallback LoRA model
                            self.character_models[character_id] = model_with_adapter
                            logger.info(f"✅ Successfully loaded fallback LoRA adapter for {character_id}")
                            
                        except Exception as fallback_e:
                            logger.error(f"❌ Even fallback LoRA failed for {character_id}: {fallback_e}")
                            logger.warning(f"Using isolated base model as final fallback")
                            self.character_models[character_id] = isolated_fallback_model
                    
                    # Cleanup temp files
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                except Exception as e1:
                    logger.warning(f"LoRA loading failed for {character_id}: {e1}")
                    
                    # Ultimate fallback: Create isolated base model for this character
                    logger.info(f"Creating ultimate fallback isolated model for {character_id}")
                    isolated_ultimate_model = AutoModelForCausalLM.from_pretrained(
                        settings.BASE_MODEL,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        device_map="cpu"
                    )
                    self.character_models[character_id] = isolated_ultimate_model
                    logger.info(f"⚠️ Using isolated base model fallback for {character_id}")
                        
            except Exception as e:
                logger.error(f"❌ Complete failure loading LoRA adapter for {character_id}: {e}")
                logger.error(f"   Adapter path: {adapter_path}")
                # Ultimate fallback to isolated base model
                final_isolated_model = AutoModelForCausalLM.from_pretrained(
                    settings.BASE_MODEL,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="cpu"
                )
                self.character_models[character_id] = final_isolated_model
                logger.info(f"⚠️ Ultimate fallback: Using isolated base model for {character_id}")
        else:
            missing_files = []
            if not os.path.exists(adapter_model_path):
                missing_files.append("adapter_model.safetensors")
            if not os.path.exists(adapter_config_path):
                missing_files.append("adapter_config.json")
            
            logger.warning(f"❌ No trained LoRA adapter found for {character_id}")
            logger.warning(f"   Missing files: {', '.join(missing_files)}")
            logger.warning(f"   Path checked: {adapter_path}")
            logger.warning(f"   Using isolated base model with character prompt only")
            missing_isolated_model = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="cpu"
            )
            self.character_models[character_id] = missing_isolated_model
            
    def _create_cache_key(self, character_id: str, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Create a cache key for response caching"""
        history_str = str(conversation_history[-2:]) if conversation_history else ""  # Only last 2 for caching
        cache_input = f"{character_id}:{user_message}:{history_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def generate_response(
        self, 
        character_id: str, 
        user_message: str, 
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate response as specific character with caching"""
        
        # Check cache first for faster responses
        cache_key = self._create_cache_key(character_id, user_message, conversation_history)
        if cache_key in self.response_cache:
            logger.info(f"Cache hit for {character_id}")
            return self.response_cache[cache_key]
        
        if character_id not in self.character_models:
            raise ValueError(f"Character {character_id} not available")
            
        # Get character-specific model and prompt - each character has their own isolated model
        model = self.character_models[character_id]
        system_prompt = self.character_prompts.get(character_id, "")
        
        # Build conversation context
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add minimal conversation history for speed
        if conversation_history:
            messages.extend(conversation_history[-2:])  # Keep only last 2 messages
            
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Format for the model
        formatted_prompt = self._format_messages(messages)
        
        # Extended tokenization for longer context and responses
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            max_length=1024,  # Much longer context for detailed responses
            truncation=True,
            padding=False
        )
        
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        
        # Character-focused generation optimized for consistency
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,  # Balanced for character consistency
                    min_new_tokens=25,   # Ensure substantial responses
                    temperature=0.7,     # Stable creativity for character voice
                    top_p=0.85,          # Focused sampling
                    top_k=40,            # Controlled variety
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15, # Stronger penalty for cleaner responses
                    use_cache=True,
                    no_repeat_ngram_size=2,   # Prevent immediate repetition
                    early_stopping=True       # Natural completion
                )
        except Exception as gen_error:
            logger.warning(f"Generation failed: {gen_error}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again!"
            
        # Decode response (skip the input tokens)
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        # Clean up response - stop at conversation indicators (updated for new format)
        stop_phrases = ["Human:", "\nHuman:", "User:", "\nUser:", "<|endoftext|>", "<|", "\n\nHuman:"]
        for stop_phrase in stop_phrases:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0].strip()
        
        # Remove meta-commentary patterns
        meta_patterns = [
            "Let me see.", "As Moses, I", "As a Samsung employee, I", "As Jinx, I", 
            "The user", "I should respond", "I need to", "Let me think",
            "Okay, the user", "I would", "Since I"
        ]
        
        for pattern in meta_patterns:
            if response.startswith(pattern):
                # Find the first sentence that doesn't contain meta-commentary
                sentences = response.split('.', 1)
                if len(sentences) > 1:
                    response = sentences[1].strip()
        
        # Clean up incomplete sentences at the end
        import re
        # If response ends mid-sentence (no punctuation), try to find last complete sentence
        if response and not response.strip()[-1] in '.!?':
            sentences = re.split(r'[.!?]+', response)
            if len(sentences) > 1:
                # Keep all complete sentences
                complete_sentences = sentences[:-1]  # Remove the incomplete last sentence
                if complete_sentences:
                    response = '. '.join(complete_sentences).strip()
                    if response and not response.endswith(('.', '!', '?')):
                        response += '.'
        
        # Ensure we have a meaningful response
        if not response or len(response.strip()) < 10:
            fallback_responses = {
                "jinx": "*spins around excitedly* Hey there! Ready for some chaos? What's cooking in that brain of yours?",
                "moses": "Peace be with you, my child. How may I guide you in the ways of the Almighty?",
                "samsung_employee": "Hello! I'm excited to help you discover amazing Samsung technology!"
            }
            response = fallback_responses.get(character_id, "Hello! How can I help you today?")
        
        # Cache the response (limit cache size)
        if len(self.response_cache) > 50:  # Simple cache size limit
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
        
        # Clean response to remove meta-commentary and character bleed
        response = self._clean_response(response, character_id)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated response for {character_id} in {generation_time:.2f}s")
        
        return response
    
    def _clean_response(self, response: str, character_id: str) -> str:
        """Clean response to remove meta-commentary and ensure character consistency"""
        if not response:
            return response
            
        import re
        
        # Remove common meta-commentary patterns
        meta_patterns = [
            r"Answer:\s*",
            r"Response:\s*",
            r"This (?:response|answer)\s.*?[.!?]",
            r"Let me (?:think|consider|analyze)\s.*?[.!?]", 
            r"Based on (?:the|this)\s.*?[.!?]",
            r"The (?:user|question)\s.*?[.!?]",
            r"I (?:need to|should|will)\s(?:respond|answer)\s.*?[.!?]",
            r"\(.*?\)",  # Remove parenthetical commentary
            r"Looking at.*?[.!?]",
            r"Analyzing.*?[.!?]",
        ]
        
        for pattern in meta_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove character name prefixes that cause bleed
        response = re.sub(r"^(?:Moses|Samsung_Employee|Jinx):\s*", "", response, flags=re.IGNORECASE)
        
        # Remove multiple character references
        other_chars = ["Moses", "Samsung_Employee", "Jinx"]
        for char in other_chars:
            if char.lower() != character_id.lower():
                response = re.sub(f"{char}:\s*", "", response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and newlines
        response = re.sub(r"\n\s*\n+", "\n\n", response)
        response = re.sub(r"\s+", " ", response)  # Multiple spaces to single
        response = response.strip()
        
        # If response is too short after cleaning, provide character-appropriate fallback
        if len(response.split()) < 5:
            fallbacks = {
                "moses": "Peace be with you, my child. How may I guide you in the ways of the Almighty?",
                "samsung_employee": "Hello! I'm excited to help you discover the amazing features of Samsung Galaxy devices!",
                "jinx": "*spins around excitedly* Hey there! Ready for some chaos? I've got explosive ideas to share!"
            }
            response = fallbacks.get(character_id, "Hello! How can I help you today?")
        
        return response
        
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for character-focused training format"""
        formatted = ""
        
        # Add conversation history
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                continue  # Skip system messages for character-focused format
            elif role == "user":
                formatted += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted += f"{content}\n\nHuman: "
                
        return formatted
        
    async def switch_character(self, character_id: str):
        """Switch to different character"""
        if character_id in self.character_models:
            self.current_character = character_id
            logger.info(f"Switched to character: {character_id}")
        else:
            raise ValueError(f"Character {character_id} not available")
            
    def get_available_characters(self) -> List[str]:
        """Get list of available character IDs"""
        return list(self.character_models.keys())
        
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Response cache cleared")
        
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.response_cache),
            "available_characters": len(self.character_models),
            "current_character": self.current_character
        }