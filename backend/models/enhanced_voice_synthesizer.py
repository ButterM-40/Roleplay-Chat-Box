import torch
import asyncio
import logging
import base64
import io
import numpy as np
from typing import Optional
from backend.config import settings
import math

logger = logging.getLogger(__name__)

class SimpleVoiceSynthesizer:
    """
    An improved simple voice synthesizer that creates more realistic speech-like audio
    using phoneme patterns, formant synthesis, and prosody modeling.
    """
    
    def __init__(self):
        self.character_voice_configs = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize simple voice synthesis"""
        if not settings.ENABLE_VOICE:
            logger.info("Voice synthesis disabled in config")
            return False
            
        logger.info("Initializing improved simple voice synthesizer...")
        
        try:
            # Setup character-specific voice parameters
            self._setup_character_voices()
            self.initialized = True
            logger.info("Improved simple voice synthesizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize simple voice synthesizer: {e}")
            return False
            
    def _setup_character_voices(self):
        """Setup character-specific voice configurations"""
        self.character_voice_configs = {
            "moses": {
                "base_frequency": 110,  # Lower, more authoritative
                "speed": 0.85,  # Slower, more measured
                "pitch_variance": 0.15,  # Less pitch variation
                "formant_shift": -0.2,  # Deeper formants
                "voice_quality": "deep",
            },
            "samsung_employee": {
                "base_frequency": 140,  # Professional, clear
                "speed": 1.0,  # Normal speed
                "pitch_variance": 0.2,  # Moderate variation
                "formant_shift": 0.0,  # Neutral formants
                "voice_quality": "clear",
            },
            "jinx": {
                "base_frequency": 180,  # Higher, more energetic
                "speed": 1.2,  # Faster speech
                "pitch_variance": 0.35,  # More pitch variation
                "formant_shift": 0.3,  # Brighter formants
                "voice_quality": "bright",
            }
        }
        
    async def synthesize(self, text: str, character_id: str) -> Optional[str]:
        """Synthesize speech for given text and character"""
        if not self.initialized or not settings.ENABLE_VOICE:
            return None
            
        try:
            # Get character voice config
            voice_config = self.character_voice_configs.get(
                character_id, 
                self.character_voice_configs["samsung_employee"]  # Default
            )
            
            # Generate realistic speech audio
            audio_data = self._generate_realistic_speech(text, voice_config)
            
            # Convert to base64 for web transmission
            audio_base64 = self._audio_to_base64(audio_data)
            
            logger.info(f"Generated realistic speech for {character_id}: {len(text)} chars, {len(audio_data)} samples")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error in simple voice synthesis: {e}")
            return None
            
    def _generate_realistic_speech(self, text: str, voice_config: dict) -> np.ndarray:
        """Generate realistic speech using advanced phoneme and prosody modeling"""
        
        # Calculate duration based on speaking rate
        words = len(text.split())
        chars = len(text)
        
        # Realistic speaking rates: 150-180 words per minute
        base_wpm = 160
        speed_factor = voice_config["speed"]
        actual_wpm = base_wpm * speed_factor
        
        # Calculate duration
        duration = (words / actual_wpm) * 60  # Convert to seconds
        duration = max(duration, chars / 15.0)  # Minimum based on character count
        duration = min(duration, 30.0)  # Maximum 30 seconds
        
        sample_rate = settings.SAMPLE_RATE
        num_samples = int(duration * sample_rate)
        
        # Create time array
        t = np.linspace(0, duration, num_samples)
        
        # Generate phoneme-based speech patterns
        audio = self._create_phoneme_speech(t, text, voice_config)
        
        # Apply prosody (intonation patterns)
        prosody = self._generate_prosody(t, text, voice_config)
        audio *= prosody
        
        # Apply character-specific voice quality
        audio = self._apply_voice_quality(audio, t, voice_config)
        
        # Add natural speech envelope
        envelope = self._create_speech_envelope(audio, t)
        audio *= envelope
        
        # Normalize and return
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
            
        return audio.astype(np.float32)
        
    def _create_phoneme_speech(self, t: np.ndarray, text: str, voice_config: dict) -> np.ndarray:
        """Create speech-like audio using phoneme patterns"""
        
        audio = np.zeros_like(t)
        base_freq = voice_config["base_frequency"]
        
        # Create syllable timing based on text
        syllable_rate = 4.0 * voice_config["speed"]  # syllables per second
        syllable_duration = 1.0 / syllable_rate
        
        for i, sample_time in enumerate(t):
            # Determine current syllable position
            syllable_phase = (sample_time % syllable_duration) / syllable_duration
            
            # Create vowel/consonant pattern
            # Vowels: 0.2-0.8 of syllable, Consonants: 0.0-0.2 and 0.8-1.0
            is_vowel = 0.2 < syllable_phase < 0.8
            
            # Get fundamental frequency with natural variation
            pitch_variation = voice_config["pitch_variance"]
            f0 = base_freq * (1 + pitch_variation * np.sin(2 * np.pi * 2.3 * sample_time))
            
            if is_vowel:
                # Generate vowel sound using formant synthesis
                vowel_sound = self._generate_vowel_formants(sample_time, f0, voice_config)
                audio[i] = vowel_sound
            else:
                # Generate consonant sound using filtered noise
                consonant_sound = self._generate_consonant(sample_time, f0, voice_config)
                audio[i] = consonant_sound
                
        return audio
        
    def _generate_vowel_formants(self, t: float, f0: float, voice_config: dict) -> float:
        """Generate vowel sounds using formant frequencies"""
        
        formant_shift = voice_config["formant_shift"]
        
        # Vowel formant frequencies (approximate average)
        f1 = 650 * (1 + formant_shift * 0.5)   # First formant
        f2 = 1400 * (1 + formant_shift * 0.3)  # Second formant
        f3 = 2500 * (1 + formant_shift * 0.2)  # Third formant
        
        # Add slight formant movement for naturalness
        f1 += 50 * np.sin(2 * np.pi * 1.7 * t)
        f2 += 80 * np.sin(2 * np.pi * 2.1 * t)
        
        # Generate harmonic series for fundamental
        fundamental = 0.4 * np.sin(2 * np.pi * f0 * t)
        
        # Generate formant resonances
        formant1 = 0.3 * np.sin(2 * np.pi * f1 * t) * np.exp(-abs(f1 - f0*1) / 200)
        formant2 = 0.2 * np.sin(2 * np.pi * f2 * t) * np.exp(-abs(f2 - f0*2) / 300)
        formant3 = 0.1 * np.sin(2 * np.pi * f3 * t) * np.exp(-abs(f3 - f0*3) / 500)
        
        # Add harmonics
        harmonic2 = 0.2 * np.sin(2 * np.pi * f0 * 2 * t)
        harmonic3 = 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
        
        return fundamental + formant1 + formant2 + formant3 + harmonic2 + harmonic3
        
    def _generate_consonant(self, t: float, f0: float, voice_config: dict) -> float:
        """Generate consonant sounds using filtered noise and fricatives"""
        
        # Create noise component for fricatives
        noise = (np.random.randn() - 0.5) * 0.15
        
        # Add some periodic component for voiced consonants
        periodic = 0.1 * np.sin(2 * np.pi * f0 * t)
        
        # Filter noise based on consonant type (simplified)
        filtered_noise = noise * (1 + 0.5 * np.sin(2 * np.pi * 3000 * t))
        
        return filtered_noise + periodic * 0.3
        
    def _generate_prosody(self, t: np.ndarray, text: str, voice_config: dict) -> np.ndarray:
        """Generate natural prosody (intonation) patterns"""
        
        prosody = np.ones_like(t)
        duration = t[-1] if len(t) > 0 else 1.0
        
        # Sentence-level intonation
        time_norm = t / duration
        
        if text.endswith('?'):
            # Question: rising intonation
            prosody *= (0.8 + 0.4 * time_norm)
        elif text.endswith('!'):
            # Exclamation: dramatic contour
            prosody *= (0.9 + 0.3 * np.sin(np.pi * time_norm) * np.exp(-time_norm))
        else:
            # Statement: natural declination
            prosody *= (1.0 - 0.2 * time_norm)
            
        # Add micro-prosody for naturalness
        prosody *= (1 + 0.05 * np.sin(2 * np.pi * 8 * t))
        
        # Character-specific prosody
        if voice_config.get("voice_quality") == "bright":
            # More animated prosody for energetic characters
            prosody *= (1 + 0.1 * np.sin(2 * np.pi * 2.5 * t))
        elif voice_config.get("voice_quality") == "deep":
            # More steady prosody for authoritative characters
            prosody *= (1 + 0.03 * np.sin(2 * np.pi * 1.2 * t))
            
        return prosody
        
    def _apply_voice_quality(self, audio: np.ndarray, t: np.ndarray, voice_config: dict) -> np.ndarray:
        """Apply character-specific voice quality effects"""
        
        quality = voice_config.get("voice_quality", "clear")
        
        if quality == "deep":
            # Add subtle sub-harmonics for deeper voice
            subharmonic = 0.05 * np.sin(np.pi * t)
            audio = audio + subharmonic[:len(audio)]
            
        elif quality == "bright":
            # Emphasize higher frequencies for brighter voice
            high_freq = 0.03 * np.sin(2 * np.pi * 4000 * t)
            audio = audio + high_freq[:len(audio)]
            
        # Add very subtle vocal fry for naturalness
        fry_rate = 70  # Hz
        fry = 0.01 * np.sin(2 * np.pi * fry_rate * t) * (np.random.randn(len(t)) * 0.5 + 0.5)
        audio = audio + fry[:len(audio)]
        
        return audio
        
    def _create_speech_envelope(self, audio: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Create natural speech amplitude envelope"""
        
        envelope = np.ones_like(audio)
        
        # Fade in/out
        fade_samples = min(int(0.05 * len(audio)), 1000)
        if fade_samples > 0:
            envelope[:fade_samples] *= np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
        # Add speech rhythm (breathing, pauses)
        breath_rate = 0.3  # Subtle breathing pattern
        envelope *= (0.95 + 0.05 * np.sin(2 * np.pi * breath_rate * t))
        
        return envelope
        
    def _audio_to_base64(self, audio_data: np.ndarray) -> str:
        """Convert audio numpy array to base64 string"""
        # Convert to 16-bit PCM
        audio_int16 = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
        
        # Create WAV file in memory manually
        buffer = io.BytesIO()
        
        # WAV file parameters
        sample_rate = settings.SAMPLE_RATE
        num_channels = 1  # Mono
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_int16) * 2  # 2 bytes per sample
        file_size = 36 + data_size
        
        # Write WAV header (44 bytes)
        buffer.write(b'RIFF')                                    # Chunk ID (4 bytes)
        buffer.write(file_size.to_bytes(4, 'little'))           # File size - 8 (4 bytes)
        buffer.write(b'WAVE')                                    # Format (4 bytes)
        buffer.write(b'fmt ')                                    # Subchunk1 ID (4 bytes)
        buffer.write((16).to_bytes(4, 'little'))                # Subchunk1 size (4 bytes)
        buffer.write((1).to_bytes(2, 'little'))                 # Audio format (PCM) (2 bytes)
        buffer.write(num_channels.to_bytes(2, 'little'))        # Num channels (2 bytes)
        buffer.write(sample_rate.to_bytes(4, 'little'))         # Sample rate (4 bytes)
        buffer.write(byte_rate.to_bytes(4, 'little'))           # Byte rate (4 bytes)
        buffer.write(block_align.to_bytes(2, 'little'))         # Block align (2 bytes)
        buffer.write(bits_per_sample.to_bytes(2, 'little'))     # Bits per sample (2 bytes)
        buffer.write(b'data')                                    # Subchunk2 ID (4 bytes)
        buffer.write(data_size.to_bytes(4, 'little'))           # Subchunk2 size (4 bytes)
        
        # Write audio data
        buffer.write(audio_int16.tobytes())
        
        # Get bytes and encode to base64
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return f"data:audio/wav;base64,{audio_base64}"
        
    def get_character_voice_info(self, character_id: str) -> dict:
        """Get voice configuration for character"""
        return self.character_voice_configs.get(character_id, {})