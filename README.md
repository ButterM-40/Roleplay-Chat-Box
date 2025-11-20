---
title: Roleplay Chat Box 🎭
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: hf_app.py
pinned: false
license: mit
short_description: Multi-character AI roleplay chatbot with LoRA-adapted personalities
tags:
- conversational-ai
- roleplay
- lora
- character-ai
- chatbot
- transformers
- qwen
models:
- Qwen/Qwen2.5-1.5B
---

# Roleplay Chat Box 🎭

An intelligent conversational AI system featuring multiple character personalities powered by LoRA (Low-Rank Adaptation) fine-tuned language models. Each character has been trained with specific datasets to maintain unique personalities, speaking patterns, and areas of expertise.

## Features

- **Multi-Character Support**: Switch between different character personalities using LoRA adapters
- **Voice Synthesis**: Text-to-speech with character-specific voices using Microsoft's VibeVoice
- **Modern Interface**: ChatGPT-style web interface with character selection sidebar
- **Real-time Chat**: WebSocket-based real-time communication
- **Audio Toggle**: Optional voice output for responses

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Models     │
│   (React/HTML)  │◄──►│   (FastAPI)     │◄──►│   Base + LoRA   │
│                 │    │                 │    │   + VibeVoice   │
│ - Chat UI       │    │ - Model Loading │    │                 │
│ - Character     │    │ - LoRA Switching│    │ - Qwen2 Base    │
│   Selection     │    │ - TTS Service   │    │ - Character     │
│ - Audio Toggle  │    │ - WebSocket     │    │   Adapters      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Characters

1. **Moses** - Biblical prophet with wisdom and authority
2. **Samsung Employee** - Tech-savvy corporate representative
3. **Jinx** - Complex, chaotic character from Arcane

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download models (see Resources section)

3. Start backend:
```bash
cd backend
python main.py
```

4. Start frontend:
```bash
cd frontend
# Instructions for your chosen frontend framework
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~8GB RAM minimum
- ~20GB storage for models

## Models Used

- Base: Qwen2 (via Polarium/qwen2-yoda-lora)
- Voice: Microsoft VibeVoice-1.5B
- Custom LoRA adapters for each character