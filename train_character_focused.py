#!/usr/bin/env python3
"""
Character-Focused LoRA Training Script
Trains character-specific adapters without Assistant/User patterns
Uses pure character embodiment approach for better roleplay consistency
"""

import os
import json
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterFocusedTrainer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", use_gpu=True):
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.base_model = None
        
        logger.info(f"🎭 Initializing Character-Focused LoRA Trainer")
        logger.info(f"📱 Model: {model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        
    def setup_model_and_tokenizer(self):
        """Load base model and tokenizer with optimal settings"""
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token properly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("🧠 Loading base model...")
        if self.device == "cuda":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
        logger.info("✅ Model and tokenizer loaded successfully!")
        
    def load_character_dataset(self, character_name):
        """Load character-focused dataset without Assistant/User patterns"""
        
        try:
            from character_focused_datasets import get_all_character_datasets
            
            datasets = get_all_character_datasets()
            
            if character_name not in datasets:
                logger.error(f"❌ Character '{character_name}' not found in datasets")
                return None
                
            logger.info(f"✨ Loading character-focused dataset for {character_name}")
            raw_data = datasets[character_name]
            
        except ImportError:
            logger.error("❌ Character-focused datasets not available")
            return None
        
        # Convert to training format - direct character responses
        training_data = []
        
        # Character-specific response starters to reinforce identity
        character_starters = {
            "moses": [
                "My child, ",
                "Listen well, ",
                "The Lord has shown me that ",
                "In my years leading Israel, ",
                "When I stood before Pharaoh, ",
                "The Almighty teaches us that "
            ],
            "samsung_employee": [
                "Samsung's technology is incredible - ",
                "I'm excited to share that ",
                "The Galaxy experience offers ",
                "Our innovation team has created ",
                "Samsung customers love how ",
                "What makes Samsung special is "
            ],
            "jinx": [
                "*bounces excitedly* ",
                "*grins maniacally* ",
                "*fidgets with braids* ",
                "*spins around* ",
                "*giggles* ",
                "*eyes light up* "
            ]
        }
        
        for item in raw_data:
            # Create simple prompt -> response format without system roles
            # This trains the model to directly embody the character
            prompt = item['prompt']
            response = item['response']
            
            # Format as a natural conversation continuation
            conversation = f"Human: {prompt}\n\n{response}<|endoftext|>"
            
            training_data.append({"text": conversation})
            
        logger.info(f"✅ Loaded {len(training_data)} character-focused examples for {character_name}")
        return Dataset.from_list(training_data)
        
    def tokenize_dataset(self, dataset):
        """Tokenize dataset with character-focused approach"""
        logger.info("🔤 Tokenizing character dataset...")
        
        def tokenize_function(examples):
            # Tokenize with focus on character responses
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,  # Shorter context to focus on character consistency
                return_overflowing_tokens=False,
            )
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Dataset tokenized: {len(tokenized_dataset)} character samples")
        return tokenized_dataset
        
    def create_character_lora_config(self):
        """Create LoRA configuration optimized for character consistency"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Lower rank for better character focus
            lora_alpha=16,  # Lower alpha for more stable training  
            lora_dropout=0.05,  # Lower dropout for consistency
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"      # MLP layers  
            ],
            bias="none",
            modules_to_save=None,
        )
        
    def train_character_adapter(self, character_name, epochs=5, batch_size=2):
        """Train character-specific LoRA adapter"""
        logger.info(f"🎭 Training character adapter for {character_name}")
        
        # Load character dataset
        dataset = self.load_character_dataset(character_name)
        if dataset is None:
            return False
            
        # Tokenize dataset  
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Prepare model for training
        if self.device == "cuda":
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            
        # Create LoRA model
        lora_config = self.create_character_lora_config()
        model = get_peft_model(self.base_model, lora_config)
        
        logger.info(f"🔧 Character LoRA model created")
        logger.info(f"📊 Trainable parameters: {model.get_nb_trainable_parameters()}")
        
        # Setup training arguments for character consistency
        output_dir = f"lora_adapters/{character_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove old adapter if exists to start fresh
        for file in os.listdir(output_dir):
            if file.endswith(('.safetensors', '.bin')):
                os.remove(os.path.join(output_dir, file))
                logger.info(f"🗑️  Removed old adapter: {file}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Accumulate for effective batch size
            learning_rate=1e-4,  # Lower learning rate for stability
            fp16=False,  # Disable for CPU compatibility
            logging_steps=10,
            save_steps=50,  # Save frequently
            save_total_limit=2,  # Keep recent checkpoints
            remove_unused_columns=False,
            dataloader_drop_last=True,
            dataloader_pin_memory=False,
            warmup_steps=20,  # Warmup for stable training
            optim="adamw_torch",
            report_to=[],  # No external logging
            load_best_model_at_end=False,
            max_grad_norm=0.5,  # Stricter gradient clipping
            logging_first_step=True,
            eval_strategy="no",  # No evaluation, just training
            save_safetensors=True,  # Use safetensors format
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info(f"🚀 Starting character training for {character_name}...")
        start_time = datetime.now()
        
        try:
            trainer.train()
            
            # Save the final model
            trainer.save_model(output_dir)
            
            # Save character-specific metadata
            metadata = {
                "character": character_name,
                "model": self.model_name,
                "device": self.device,
                "epochs": epochs,
                "batch_size": batch_size,
                "dataset_size": len(dataset),
                "training_approach": "character_focused",
                "training_time": str(datetime.now() - start_time),
                "timestamp": datetime.now().isoformat(),
                "lora_config": {
                    "r": lora_config.r,
                    "alpha": lora_config.lora_alpha,
                    "dropout": lora_config.lora_dropout,
                    "target_modules": lora_config.target_modules
                }
            }
            
            with open(f"{output_dir}/training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"✅ Character training completed for {character_name}!")
            logger.info(f"⏱️  Training time: {datetime.now() - start_time}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Character training failed for {character_name}: {e}")
            return False
            
    def train_all_characters(self, epochs=5, batch_size=2):
        """Train character adapters for all characters"""
        characters = ["moses", "samsung_employee", "jinx"]
        results = {}
        
        logger.info(f"🎭 Training character adapters for all characters")
        logger.info(f"📊 Configuration: {epochs} epochs, batch size {batch_size}")
        
        total_start_time = datetime.now()
        
        for character in characters:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 TRAINING CHARACTER: {character.upper()}")
            logger.info(f"{'='*60}")
            
            success = self.train_character_adapter(character, epochs, batch_size)
            results[character] = success
            
            if success:
                logger.info(f"✅ {character} character training: SUCCESS")
            else:
                logger.info(f"❌ {character} character training: FAILED")
                
        total_time = datetime.now() - total_start_time
        
        # Training summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 CHARACTER TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Total time: {total_time}")
        
        successful = sum(results.values())
        logger.info(f"✅ Successful characters: {successful}/{len(characters)}")
        
        for character, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"   {character}: {status}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Train character-focused LoRA adapters")
    parser.add_argument("--character", type=str, help="Character to train (moses, samsung_employee, jinx)")
    parser.add_argument("--all", action="store_true", help="Train all characters")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size") 
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Initialize character trainer
    trainer = CharacterFocusedTrainer(
        model_name="Qwen/Qwen3-0.6B",
        use_gpu=not args.cpu
    )
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    if args.all:
        # Train all characters
        results = trainer.train_all_characters(args.epochs, args.batch_size)
        
        if all(results.values()):
            print("\n🎉 All character adapters trained successfully!")
        else:
            print("\n⚠️  Some character trainings failed. Check logs above.")
            
    elif args.character:
        # Train specific character
        if args.character in ["moses", "samsung_employee", "jinx"]:
            success = trainer.train_character_adapter(
                args.character, 
                args.epochs, 
                args.batch_size
            )
            
            if success:
                print(f"\n🎉 {args.character} character trained successfully!")
            else:
                print(f"\n❌ {args.character} character training failed!")
        else:
            print("❌ Invalid character. Choose: moses, samsung_employee, jinx")
            
    else:
        print("❌ Please specify --character <name> or --all")
        
if __name__ == "__main__":
    main()