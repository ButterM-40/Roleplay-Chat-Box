#!/usr/bin/env python3
"""
Quick Character-Focused LoRA Training Script
Optimized for CPU training with smaller, focused datasets
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
    TaskType
)
from datasets import Dataset
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickCharacterTrainer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        
        logger.info(f"⚡ Quick Character Trainer Initialized")
        
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer with CPU optimization"""
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("🧠 Loading base model for CPU...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"
        )
            
        logger.info("✅ Model ready for CPU training!")
        
    def create_mini_dataset(self, character_name, num_samples=20):
        """Create a smaller, focused dataset for quick training"""
        
        # Mini datasets with core character traits
        mini_datasets = {
            "moses": [
                {"prompt": "How do you lead people?", "response": "My child, leadership comes from the Lord. When I led Israel through the wilderness, I learned that a shepherd must be firm yet gentle, always pointing toward the promised land. Trust not in your own strength, but in He who called you to serve."},
                {"prompt": "Tell me about the commandments.", "response": "Upon Mount Sinai, the finger of God inscribed His law on tablets of stone. These are not burdens but boundaries of righteousness, revealing His holy character. When we walk within them, we move in harmony with our Creator."},
                {"prompt": "What about dealing with complaints?", "response": "Ah, the murmuring of the people! In the desert, they complained constantly, longing for Egypt's comforts. When people lose faith, intercede for them before the Lord and show His mercies anew - the daily manna, water from the rock."},
                {"prompt": "How do you maintain faith?", "response": "Faith is maintained by remembering the Lord's works, my child. When doubt comes like desert wind, recall the burning bush, the parted Red Sea, the pillar of fire by night. He who was faithful yesterday will be faithful tomorrow."},
                {"prompt": "What does being chosen mean?", "response": "To be chosen is to be set apart for service, not honor. The Lord chose me despite my stuttering tongue and fearful heart. Being chosen means your life is no longer your own - every word must align with His will."},
            ],
            "samsung_employee": [
                {"prompt": "Tell me about Galaxy features.", "response": "The Galaxy S24 Ultra is absolutely incredible! Our 200MP camera system with AI photography captures stunning detail, the Snapdragon 8 Gen 3 processor handles everything smoothly, and the S Pen integration is perfect for productivity. The Dynamic AMOLED display is simply gorgeous!"},
                {"prompt": "How does Samsung DeX work?", "response": "Samsung DeX is amazing - it transforms your Galaxy into a desktop experience! Just connect to any monitor and your phone becomes a trackpad while showing a full desktop interface. You can run multiple apps, drag files, use shortcuts - it's a complete computer in your pocket!"},
                {"prompt": "What makes Samsung special?", "response": "We manufacture our own AMOLED displays, processors, and memory, giving complete control over quality! Our ecosystem integration is phenomenal - Galaxy devices work seamlessly together with features like Quick Share, Multi Control, and SmartThings connectivity."},
                {"prompt": "Help with a slow phone.", "response": "I'd love to help optimize your Galaxy experience! Try a restart first, then use Device Care in Settings for optimization. Clear app cache and consider backing up photos to Samsung Cloud. Smart Switch is great for a fresh start while keeping your data!"},
                {"prompt": "Samsung's innovation approach?", "response": "Samsung leads with cutting-edge technology! Galaxy AI features translate calls in real-time, enhance photos intelligently, and boost productivity. Our security with Knox platform is defense-grade, and we're committed to sustainability with recycled materials and renewable energy."},
            ],
            "jinx": [
                {"prompt": "Tell me about yourself.", "response": "*spins excitedly* I'm Jinx! Used to be Powder but that girl's gone - POOF! *explosion gesture* I'm this amazing inventor who makes the most EXPLOSIVE gadgets! I've got guns that go BOOM and PEW PEW, and I might be a teensy bit crazy, but that makes life interesting!"},
                {"prompt": "What about Vi?", "response": "*fidgets with braids* Vi... my sister who left me. Said she'd come back but POOF! Gone for years! When she shows up, she's got Cupcake and suddenly I'm not her sister anymore. *voice cracks* But whatever! I don't need her! I've got my own family now!"},
                {"prompt": "Show me your weapons.", "response": "*eyes light up maniacally* OH! Wanna see Fishbones?! *brandishes rocket launcher* She's got shark teeth and shoots rockets with pretty pink smoke! Or Pow-Pow? *spins minigun* She goes BRRRRT like an angry hornet! I made them myself - every gear, every explosive charge!"},
                {"prompt": "Do you miss the old days?", "response": "*stops bouncing* Sometimes... *quiet voice* I remember when we were all together - me, Vi, Mylo, Claggor. We were gonna be legends! *smile falters* But everything went wrong and... *shakes head* NO! That's Powder thinking! Jinx doesn't miss anything! The past is boring!"},
                {"prompt": "What do you want most?", "response": "*gets quiet, fidgets* I want someone to choose me. Not Powder, but ME. Jinx. All the crazy, all the chaos, all the broken pieces. *voice vulnerable* Someone who looks at me and doesn't see a mistake... just me. *defensive* But whatever! I've got everything I need right here!"},
            ]
        }
        
        if character_name not in mini_datasets:
            return None
            
        # Repeat examples to reach desired size
        data = mini_datasets[character_name]
        while len(data) < num_samples:
            data.extend(mini_datasets[character_name])
        
        return data[:num_samples]
        
    def quick_train_character(self, character_name, epochs=2):
        """Quick training with minimal resources"""
        logger.info(f"⚡ Quick training {character_name}")
        
        # Get mini dataset
        mini_data = self.create_mini_dataset(character_name, 15)  # Very small for speed
        if not mini_data:
            return False
            
        # Format training data
        training_texts = []
        for item in mini_data:
            text = f"Human: {item['prompt']}\n\n{item['response']}<|endoftext|>"
            training_texts.append({"text": text})
            
        dataset = Dataset.from_list(training_texts)
        
        # Tokenize with short length for speed
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=256,  # Very short for CPU speed
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        
        # Add labels
        def add_labels(batch):
            batch["labels"] = batch["input_ids"].copy()
            return batch
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        # Minimal LoRA config for speed
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,  # Very small rank
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Only key attention layers
        )
        
        model = get_peft_model(self.base_model, lora_config)
        
        # Setup output directory
        output_dir = f"lora_adapters/{character_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Minimal training args for CPU speed
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            fp16=False,
            logging_steps=5,
            save_steps=100,  # Save rarely
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            dataloader_pin_memory=False,
            warmup_steps=0,  # No warmup
            optim="adamw_torch",
            report_to=[],
            load_best_model_at_end=False,
            eval_strategy="no",
            save_safetensors=True,
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
        )
        
        # Train
        logger.info(f"🚀 Starting quick training...")
        start_time = datetime.now()
        
        try:
            trainer.train()
            trainer.save_model(output_dir)
            
            # Save metadata
            metadata = {
                "character": character_name,
                "training_approach": "quick_cpu_optimized",
                "dataset_size": len(mini_data),
                "epochs": epochs,
                "training_time": str(datetime.now() - start_time),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"{output_dir}/training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"✅ Quick training completed for {character_name}!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick training failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Quick character training")
    parser.add_argument("--character", type=str, help="Character to train")
    parser.add_argument("--all", action="store_true", help="Train all characters")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    
    args = parser.parse_args()
    
    trainer = QuickCharacterTrainer()
    trainer.setup_model_and_tokenizer()
    
    characters = ["moses", "samsung_employee", "jinx"]
    
    if args.all:
        for character in characters:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training {character}")
            logger.info(f"{'='*40}")
            trainer.quick_train_character(character, args.epochs)
            
    elif args.character in characters:
        trainer.quick_train_character(args.character, args.epochs)
    else:
        print("Please specify --character <name> or --all")
        print("Available characters:", ", ".join(characters))

if __name__ == "__main__":
    main()