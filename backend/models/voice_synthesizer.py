import torch
import asyncio
import logging
import base64
import io
import numpy as np
from typing import Optional
from backend.config import settings
try:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    def __init__(self):
        self.voice_model = None
        self.voice_processor = None
        self.character_voice_configs = {}
        
    async def initialize(self):
        """Initialize voice synthesis model"""
        if not settings.ENABLE_VOICE:
            logger.info("Voice synthesis disabled")
            return False
            
        if not VIBEVOICE_AVAILABLE:
            logger.error("VibeVoice community package not available. Install with: pip install git+https://github.com/vibevoice-community/VibeVoice.git")
            return False
            
        logger.info("Loading VibeVoice model...")
        
        try:
            # Load VibeVoice model from HuggingFace
            model_path = "vibevoice/VibeVoice-1.5B"
            
            # Load processor
            logger.info(f"Loading processor from {model_path}")
            self.voice_processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"
            
            logger.info(f"Loading model with device: {device}, dtype: {load_dtype}, attention: {attn_impl}")
            
            # Load model
            if device == "cuda":
                self.voice_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:
                self.voice_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
            
            # Set inference steps
            self.voice_model.eval()
            self.voice_model.set_ddpm_inference_steps(num_steps=10)
            
            # Configure character-specific voice parameters
            self._setup_character_voices()
            
            logger.info("VibeVoice synthesizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VibeVoice model: {e}")
            logger.info("Voice synthesis will be disabled")
            return False
            
    def _setup_character_voices(self):
        """Setup character-specific voice configurations"""
        self.character_voice_configs = {
            "moses": {
                "style": "authoritative",
                "speed": 0.9,  # Slightly slower, more measured
                "pitch": 0.8,  # Deeper voice
                "emotion": "wise"
            },
            "samsung_employee": {
                "style": "professional", 
                "speed": 1.0,  # Normal speed
                "pitch": 1.0,  # Normal pitch
                "emotion": "friendly"
            },
            "jinx": {
                "style": "energetic",
                "speed": 1.2,  # Faster, more manic
                "pitch": 1.3,  # Higher pitch
                "emotion": "playful"
            }
        }
        
    async def synthesize(self, text: str, character_id: str) -> Optional[str]:
        """Synthesize speech for given text and character"""
        if not settings.ENABLE_VOICE or not self.voice_model or not self.voice_tokenizer:
            return None
            
        try:
            # Get character voice config
            voice_config = self.character_voice_configs.get(
                character_id, 
                self.character_voice_configs["samsung_employee"]  # Default
            )
            
            # Prepare text for TTS
            processed_text = self._preprocess_text(text, character_id)
            
            # Process text with VibeVoice tokenizer
            inputs = self.voice_tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate audio using VibeVoice
            with torch.no_grad():
                outputs = self.voice_model.generate(
                    **inputs,
                    max_length=1024,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.8
                )
                
                # Convert outputs to audio waveform
                audio_features = outputs
            
            # Convert model outputs to audio waveform
            audio_np = self._features_to_audio(audio_features, voice_config)
            
            # Apply character-specific modifications
            audio_np = self._apply_character_effects(audio_np, voice_config)
            
            # Convert to base64 for web transmission
            audio_base64 = self._audio_to_base64(audio_np)
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error in voice synthesis: {e}")
            return None
            
    def _preprocess_text(self, text: str, character_id: str) -> str:
        """Preprocess text for character-specific speech patterns"""
        
        # Character-specific text modifications
        if character_id == "moses":
            # Add pauses for emphasis, make more formal
            text = text.replace("!", ".")  # Less exclamatory
            text = text.replace("...", "... ") # Add pauses
            
        elif character_id == "jinx":
            # Make more energetic and expressive
            text = text.replace(".", "!")  # More excitement
            text = text.replace(",", "... ") # Add dramatic pauses
            
        # Clean up text
        text = text.strip()
        
        # Add character voice prompt for better synthesis
        voice_prompts = {
            "moses": f"[Speaking with wisdom and authority] {text}",
            "samsung_employee": f"[Speaking professionally and clearly] {text}",
            "jinx": f"[Speaking energetically and playfully] {text}"
        }
        
        return voice_prompts.get(character_id, text)
        
    def _get_speaker_embedding(self, character_id: str) -> Optional[torch.Tensor]:
        """Get speaker embedding for character (simplified approach)"""
        # Create different speaker embeddings for different characters
        # This is a simplified approach - in practice, you'd train specific embeddings
        
        embeddings = {
            "moses": torch.randn(1, 512) * 0.1,  # Deeper, more authoritative
            "samsung_employee": torch.randn(1, 512) * 0.05,  # Neutral, professional
            "jinx": torch.randn(1, 512) * 0.15,  # More varied, energetic
        }
        
        # Set seed for consistency
        torch.manual_seed(hash(character_id) % 10000)
        embedding = embeddings.get(character_id, embeddings["samsung_employee"])
        
        return embedding
        
    def _spectrogram_to_audio(self, spectrogram: torch.Tensor, voice_config: dict) -> np.ndarray:
        """Convert spectrogram to audio waveform (fallback method)"""
        # This is a simplified conversion for when vocoder is not available
        
        if spectrogram.is_cuda:
            spectrogram = spectrogram.cpu()
        spec_np = spectrogram.squeeze().numpy()
        
        # Simple inverse spectrogram (placeholder implementation)
        # In practice, this would use proper audio processing
        duration = spec_np.shape[1] * 0.05  # Estimate duration
        samples = int(duration * settings.SAMPLE_RATE)
        
        # Generate audio based on spectral features
        audio = np.zeros(samples)
        for i in range(min(spec_np.shape[0], samples)):
            if i < len(audio):
                audio[i] = np.mean(spec_np[:, i % spec_np.shape[1]]) * 0.3
                
        return audio.astype(np.float32)
        
    def _apply_character_effects(self, audio: np.ndarray, voice_config: dict) -> np.ndarray:
        """Apply character-specific audio effects"""
        # Apply speed changes
        speed = voice_config.get("speed", 1.0)
        if speed != 1.0:
            audio = self._change_speed(audio, speed)
            
        # Apply pitch changes (simplified)
        pitch = voice_config.get("pitch", 1.0)
        if pitch != 1.0:
            audio = self._change_pitch(audio, pitch)
            
        return audio
        
    def _change_pitch(self, audio: np.ndarray, pitch_factor: float) -> np.ndarray:
        """Change pitch of audio (simplified implementation)"""
        if pitch_factor == 1.0:
            return audio
            
        # Simple pitch shifting by resampling (not perfect but functional)
        new_length = int(len(audio) / pitch_factor)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
        

        
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Change audio playback speed"""
        if speed == 1.0:
            return audio
            
        # Simple time stretching (placeholder)
        new_length = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
        
    def _audio_to_base64(self, audio_data: np.ndarray) -> str:
        """Convert audio numpy array to base64 string"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        
        # Write WAV header and data
        torchaudio.save(
            buffer, 
            torch.from_numpy(audio_int16).unsqueeze(0).float() / 32767.0,
            settings.SAMPLE_RATE,
            format="wav"
        )
        
        # Get bytes and encode to base64
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return f"data:audio/wav;base64,{audio_base64}"
        
    def get_character_voice_info(self, character_id: str) -> dict:
        """Get voice configuration for character"""
        return self.character_voice_configs.get(character_id, {})