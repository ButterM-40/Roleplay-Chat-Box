# Multi-Character Roleplay Chatbot Implementation Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (HTML/CSS/JS)                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  Character      │  │   Chat Interface │  │   Voice Controls    │ │
│  │  Selection      │  │   - Messages     │  │   - Toggle Audio    │ │
│  │  - Moses        │  │   - Input Box    │  │   - Play/Stop       │ │
│  │  - Samsung Emp  │  │   - Send Button  │  │   - Volume Control  │ │
│  │  - Jinx         │  │   - Typing Ind.  │  │                     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │ WebSocket
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend API (FastAPI)                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │   WebSocket     │  │  Character       │  │   Voice Synthesis   │ │
│  │   Handler       │  │  Manager         │  │   Service           │ │
│  │   - Real-time   │  │  - Model Loading │  │   - VibeVoice       │ │
│  │   - Chat Msgs   │  │  - LoRA Adapt.   │  │   - Audio Gen       │ │
│  │   - Char Switch │  │  - Response Gen  │  │   - Character Voice │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           ML Models Layer                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Base Language Model                         │   │
│  │              (Qwen2-7B-Instruct)                          │   │
│  │                                                             │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │   │
│  │  │    Moses    │ │   Samsung   │ │       Jinx          │  │   │
│  │  │ LoRA Adapter│ │ LoRA Adapter│ │   LoRA Adapter      │  │   │
│  │  │  r=16       │ │   r=16      │ │     r=16            │  │   │
│  │  │ α=32        │ │  α=32       │ │    α=32             │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Text-to-Speech Model                          │   │
│  │              (Microsoft VibeVoice)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Environment Setup

```bash
# Clone/setup project directory
cd "d:/Web Development/ChatbOx/Roleplay-Chat-Box"

# Install Python dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### Step 2: Model Preparation

**Download Base Model:**
- Primary: [Polarium/qwen2-yoda-lora](https://huggingface.co/Polarium/qwen2-yoda-lora)
- Alternative: microsoft/DialoGPT-medium (for lower hardware requirements)

**Download Voice Model:**
- Primary: [Microsoft VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)
- Alternative: microsoft/speecht5_tts

### Step 3: LoRA Adapter Training

```bash
# Train all character adapters
python train_lora_adapters.py --character all

# Or train individual characters
python train_lora_adapters.py --character moses
python train_lora_adapters.py --character samsung_employee
python train_lora_adapters.py --character jinx
```

### Step 4: Backend Deployment

```bash
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

### Step 5: Frontend Access

Open your browser to `http://localhost:8000`

## LoRA Implementation Details

### Why LoRA?

1. **Memory Efficiency**: Reduces trainable parameters by ~99%
2. **Fast Switching**: Change character personality instantly
3. **Modular Design**: Add new characters without retraining base model
4. **Cost Effective**: Fine-tune on consumer hardware

### LoRA Configuration

```python
LoraConfig(
    r=16,              # Rank (smaller = fewer parameters)
    lora_alpha=32,     # Scaling factor (typically 2x rank)
    target_modules=[   # Which model layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,  # Regularization
    task_type=TaskType.CAUSAL_LM
)
```

### Character Personalities

**Moses:**
- Wise, authoritative, spiritual
- References biblical experiences
- Speaks with measured dignity
- Focus on moral guidance

**Samsung Employee:**
- Professional, knowledgeable, enthusiastic
- Deep product expertise
- Customer-service oriented
- Technical but accessible

**Jinx:**
- Chaotic, brilliant, emotionally complex
- Rapid mood swings
- Technical genius with explosives
- Traumatic past references

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB available space
- **GPU**: CUDA-compatible (optional but recommended)
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 32GB+
- **GPU**: RTX 3080 or better with 10GB+ VRAM
- **Storage**: SSD with 50GB+ available

### Dependencies

```txt
Core ML Libraries:
- torch>=2.0.0
- transformers>=4.35.0
- peft>=0.7.0
- accelerate>=0.25.0

Backend:
- fastapi>=0.104.0
- uvicorn>=0.24.0
- websockets>=12.0

Audio Processing:
- torchaudio>=2.1.0
- soundfile>=0.12.1
```

## Usage Instructions

### 1. Character Selection
- Click character cards in the left sidebar
- Each character has unique personality and knowledge
- Switching preserves conversation context

### 2. Chat Interface
- Type messages in the input box
- Press Enter or click send button
- Responses appear with character-specific styling

### 3. Voice Features
- Toggle voice output with microphone button
- Character-specific voice parameters
- Real-time audio generation

### 4. Settings
- Adjust response speed and temperature
- Enable/disable voice synthesis
- Change UI theme (dark/light)

## Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure sufficient RAM/VRAM
- Check CUDA installation for GPU acceleration
- Verify model files downloaded correctly

**WebSocket Connection Issues:**
- Check firewall settings
- Ensure backend is running on correct port
- Verify network connectivity

**Voice Synthesis Problems:**
- Check audio device settings
- Verify VibeVoice model installation
- Test with voice toggle disabled

### Performance Optimization

**Memory Usage:**
- Use quantization (int8/int4) for large models
- Implement model offloading to CPU when idle
- Clear conversation history periodically

**Response Speed:**
- Enable GPU acceleration
- Use smaller base models for faster inference
- Implement response caching

## Extending the System

### Adding New Characters

1. **Create Character Data:**
```python
new_character = {
    "conversations": [
        {
            "input": "Sample question",
            "output": "Character-specific response"
        }
    ]
}
```

2. **Train LoRA Adapter:**
```bash
python train_lora_adapters.py --character new_character
```

3. **Update Frontend:**
- Add character card to sidebar
- Include avatar image
- Update character switching logic

4. **Update Backend:**
- Add character to available characters list
- Include character prompts and voice config

### API Extensions

**New Endpoints:**
- `/api/characters/{id}/history` - Get conversation history
- `/api/voice/synthesize` - Generate audio independently  
- `/api/models/switch` - Change base model
- `/api/adapters/train` - Train new adapters

## Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Production settings
DEBUG=False
API_HOST=0.0.0.0
API_PORT=8000

# Model paths
MODEL_PATH=/app/models
LORA_ADAPTERS_PATH=/app/lora_adapters

# Performance
DEVICE=cuda
MAX_LENGTH=2048
BATCH_SIZE=1
```

### Scaling Considerations

- **Load Balancing**: Multiple backend instances
- **Model Caching**: Redis for response caching  
- **Database**: PostgreSQL for conversation storage
- **CDN**: Static asset delivery
- **Monitoring**: Prometheus + Grafana

## Resources & Links

### Model Sources
- [Qwen2 Base Model](https://huggingface.co/Polarium/qwen2-yoda-lora)
- [Microsoft VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [PEFT Documentation](https://huggingface.co/docs/peft/en/index)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Technical References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
- [WebSocket Implementation](https://websockets.readthedocs.io/)
- [PyTorch Audio](https://pytorch.org/audio/stable/index.html)

### Character Resources
- Moses: Biblical references and historical context
- Samsung: Product manuals and technical specifications
- Jinx: Arcane series character analysis and lore