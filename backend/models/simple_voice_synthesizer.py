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
    A simple voice synthesizer that creates synthetic speech using basic audio generation.
    This is a fallback solution when VibeVoice is not available.
    """
    
    def __init__(self):
        self.character_voice_configs = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize simple voice synthesis"""
        if not settings.ENABLE_VOICE:
            logger.info("Voice synthesis disabled in config")
            return False
            
        logger.info("Initializing simple voice synthesizer...")
        
        try:
            # Setup character-specific voice parameters
            self._setup_character_voices()
            self.initialized = True
            logger.info("Simple voice synthesizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize simple voice synthesizer: {e}")
            return False
            
    def _setup_character_voices(self):
        """Setup character-specific voice configurations"""
        self.character_voice_configs = {
            "moses": {
                "base_frequency": 120,  # Lower, more authoritative
                "speed": 0.9,  # Slightly slower
                "vibrato_rate": 4.5,  # Gentle vibrato
                "vibrato_depth": 0.02,
                "formant_shift": -0.1,  # Deeper formants
            },
            "samsung_employee": {
                "base_frequency": 150,  # Professional, clear
                "speed": 1.0,  # Normal speed
                "vibrato_rate": 5.0,
                "vibrato_depth": 0.015,
                "formant_shift": 0.0,  # Neutral formants
            },
            "jinx": {
                "base_frequency": 180,  # Higher, more energetic
                "speed": 1.15,  # Faster speech
                "vibrato_rate": 6.0,  # More vibrato
                "vibrato_depth": 0.03,
                "formant_shift": 0.2,  # Brighter formants
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
            
            # Generate audio
            audio_data = self._generate_speech(text, voice_config)
            
            # Convert to base64 for web transmission
            audio_base64 = self._audio_to_base64(audio_data)
            
            logger.info(f"Generated speech for {character_id}: {len(text)} chars, audio: {len(audio_data)} samples, base64: {len(audio_base64)} chars")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error in simple voice synthesis: {e}")
            return None
            
    def _generate_speech(self, text: str, voice_config: dict) -> np.ndarray:
        """Generate synthetic speech using formant synthesis"""
        
        # Estimate duration based on text length and speech rate
        words = len(text.split())
        chars = len(text)
        
        # Rough estimation: 3-5 words per second, adjusted by speed
        base_duration = max(words / 4.0, chars / 15.0)  # Minimum based on character count
        duration = base_duration / voice_config["speed"]
        duration = min(duration, 30.0)  # Max 30 seconds
        
        sample_rate = settings.SAMPLE_RATE
        num_samples = int(duration * sample_rate)
        
        # Generate time array
        t = np.linspace(0, duration, num_samples)
        
        # Base frequency with subtle variation
        base_freq = voice_config["base_frequency"]
        
        # Add prosody (pitch contours for natural speech)
        prosody = self._generate_prosody(t, text, voice_config)
        frequency = base_freq * prosody
        
        # Add vibrato
        vibrato_rate = voice_config["vibrato_rate"]
        vibrato_depth = voice_config["vibrato_depth"]
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        frequency *= vibrato
        
        # Generate formants (multiple resonant frequencies)
        audio = self._generate_formants(t, frequency, voice_config)
        
        # Add speech-like envelope
        envelope = self._generate_envelope(t, text, voice_config)
        audio *= envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.7
            
        return audio.astype(np.float32)
        
    def _generate_prosody(self, t: np.ndarray, text: str, voice_config: dict) -> np.ndarray:
        """Generate pitch contours for natural-sounding speech"""
        
        # Basic prosody pattern
        prosody = np.ones_like(t)
        sentence_length = len(t)
        
        # Estimate word boundaries based on text length and spaces
        word_count = len(text.split())
        words_per_second = 3.0  # Average speech rate
        
        # Create word-level pitch variations
        if word_count > 1:
            word_rate = word_count / (len(t) / settings.SAMPLE_RATE)
            word_stress = 1 + 0.15 * np.sin(2 * np.pi * word_rate * t / word_count)
            prosody *= word_stress
        
        # Add sentence-level intonation based on punctuation
        time_norm = np.linspace(0, 1, sentence_length)
        
        if text.endswith('?'):
            # Question: rising intonation (more pronounced)
            prosody *= (1 + 0.3 * time_norm)
            
        elif text.endswith('!'):
            # Exclamation: dramatic rise and fall
            prosody *= (1 + 0.4 * np.sin(1.2 * np.pi * time_norm))
            
        else:
            # Statement: natural fall with slight initial rise
            prosody *= (1 + 0.2 * np.sin(np.pi * time_norm) * np.exp(-1.5 * time_norm))
        
        # Add micro-variations for naturalness
        micro_variations = 1 + 0.03 * np.sin(2 * np.pi * 12 * t)  # 12 Hz micro-variations
        prosody *= micro_variations
        
        # Character-specific prosody adjustments
        character_factor = voice_config.get("pitch", 1.0)
        if character_factor > 1.2:  # High-pitched characters (like Jinx)
            # Add more dramatic pitch swings
            prosody *= (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
        elif character_factor < 0.9:  # Low-pitched characters (like Moses)
            # More steady, authoritative prosody
            prosody *= (1 + 0.05 * np.sin(2 * np.pi * 1.5 * t))
        
        return prosody
        
    def _generate_formants(self, t: np.ndarray, frequency: np.ndarray, voice_config: dict) -> np.ndarray:
        """Generate realistic speech using formant synthesis and phoneme patterns"""
        
        # Generate phase for continuous frequency changes
        phase = np.zeros_like(t)
        for i in range(1, len(t)):
            phase[i] = phase[i-1] + 2 * np.pi * frequency[i] / settings.SAMPLE_RATE
            
        # Create voiced/unvoiced pattern based on text characteristics
        voiced_pattern = self._create_phoneme_pattern(t)
        
        # Generate rich harmonic content for voiced sounds
        voiced_audio = np.zeros_like(t)
        for i, is_voiced in enumerate(voiced_pattern):
            if is_voiced > 0.5:  # Voiced segments
                # Create rich harmonic series (like vocal cords)
                sample = 0
                for harmonic in range(1, 12):
                    if frequency[i] * harmonic < settings.SAMPLE_RATE / 2:  # Avoid aliasing
                        # Natural harmonic amplitude rolloff
                        amplitude = 0.6 / (harmonic ** 0.8) * is_voiced
                        # Add slight randomness to harmonics
                        phase_noise = 0.1 * np.sin(2 * np.pi * 7 * t[i])
                        sample += amplitude * np.sin(harmonic * phase[i] + phase_noise)
                voiced_audio[i] = sample
        
        # Apply formant filtering for vowel-like quality
        formant_shift = voice_config.get("formant_shift", 0.0)
        
        # Dynamic vowel simulation
        vowel_rate = 3.0  # Vowel changes per second
        vowel_pattern = np.sin(2 * np.pi * vowel_rate * t)
        
        # Multiple vowel formant sets (approximating /a/, /e/, /i/, /o/, /u/)
        vowel_formants = {
            'a': (730, 1090, 2440),   # /a/ as in "father"
            'e': (530, 1840, 2480),   # /e/ as in "bed" 
            'i': (270, 2290, 3010),   # /i/ as in "beat"
            'o': (570, 840, 2410),    # /o/ as in "boat"
            'u': (440, 1020, 2240)    # /u/ as in "boot"
        }
        
        # Interpolate between vowels over time
        vowel_keys = list(vowel_formants.keys())
        vowel_index = ((vowel_pattern + 1) / 2) * (len(vowel_keys) - 1)
        
        # Apply formant filtering
        filtered_audio = np.zeros_like(voiced_audio)
        
        for i in range(len(t)):
            # Get current vowel formants by interpolation
            idx = int(vowel_index[i])
            frac = vowel_index[i] - idx
            
            if idx < len(vowel_keys) - 1:
                f1_a, f2_a, f3_a = vowel_formants[vowel_keys[idx]]
                f1_b, f2_b, f3_b = vowel_formants[vowel_keys[idx + 1]]
                
                f1 = f1_a + (f1_b - f1_a) * frac
                f2 = f2_a + (f2_b - f2_a) * frac  
                f3 = f3_a + (f3_b - f3_a) * frac
            else:
                f1, f2, f3 = vowel_formants[vowel_keys[-1]]
            
            # Apply character-specific formant shift
            f1 *= (1 + formant_shift * 0.3)
            f2 *= (1 + formant_shift * 0.2) 
            f3 *= (1 + formant_shift * 0.1)
            
            # Simple formant filtering using resonance approximation
            if voiced_pattern[i] > 0.1:
                # Emphasize frequencies near formants
                sample = voiced_audio[i]
                
                # F1 resonance
                f1_resonance = 1 + 0.4 * np.exp(-((frequency[i] - f1) / 80) ** 2)
                # F2 resonance  
                f2_resonance = 1 + 0.3 * np.exp(-((frequency[i] - f2) / 120) ** 2)
                # F3 resonance
                f3_resonance = 1 + 0.2 * np.exp(-((frequency[i] - f3) / 200) ** 2)
                
                filtered_audio[i] = sample * f1_resonance * f2_resonance * f3_resonance
            else:
                # Unvoiced segments - add fricative noise
                np.random.seed(int(t[i] * 1000) % 10000)
                noise_amp = (1 - voiced_pattern[i]) * 0.15
                filtered_audio[i] = (np.random.random() - 0.5) * noise_amp
        
        return filtered_audio
    
    def _create_phoneme_pattern(self, t: np.ndarray) -> np.ndarray:
        """Create a pattern of voiced/unvoiced segments to simulate phonemes"""
        
        pattern = np.ones_like(t)
        
        # Create syllable-like rhythm
        syllable_rate = 4.5  # Syllables per second
        syllable_phase = 2 * np.pi * syllable_rate * t
        
        # Most of syllable is voiced (vowel), with brief unvoiced parts (consonants)
        voiced_base = 0.8 + 0.2 * np.sin(syllable_phase)
        
        # Add consonant-like unvoiced segments
        consonant_rate = 8.0  # Consonant events per second
        consonant_phase = 2 * np.pi * consonant_rate * t
        consonant_trigger = np.sin(consonant_phase + np.pi/4)
        
        # Sharp consonant transitions
        consonant_mask = (consonant_trigger > 0.85).astype(float)
        
        # Combine patterns - consonants reduce voicing
        pattern = voiced_base * (1 - consonant_mask * 0.7)
        
        # Smooth transitions to avoid clicks
        kernel_size = max(3, len(pattern) // 200)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if kernel_size >= 3 and kernel_size <= len(pattern) // 3:
            kernel = np.ones(kernel_size) / kernel_size
            pattern = np.convolve(pattern, kernel, mode='same')
        
        return np.clip(pattern, 0, 1)
        
    def _generate_envelope(self, t: np.ndarray, text: str, voice_config: dict) -> np.ndarray:
        """Generate amplitude envelope for speech-like rhythm"""
        
        envelope = np.ones_like(t)
        
        # Overall fade in/out
        fade_samples = min(int(0.05 * len(t)), 500)  # 50ms fade
        if fade_samples > 0:
            # Smooth fade in
            envelope[:fade_samples] *= np.sin(np.pi * np.linspace(0, 0.5, fade_samples)) ** 2
            # Smooth fade out
            envelope[-fade_samples:] *= np.cos(np.pi * np.linspace(0, 0.5, fade_samples)) ** 2
        
        # Estimate syllables from text length
        syllable_count = max(len(text.replace(' ', '')) // 3, 1)  # Rough syllable estimate
        duration = len(t) / settings.SAMPLE_RATE
        syllable_rate = syllable_count / duration
        
        # Create syllable-like amplitude modulation
        syllable_pattern = 0.6 + 0.4 * (np.sin(2 * np.pi * syllable_rate * t) ** 2)
        envelope *= syllable_pattern
        
        # Add word boundaries (pauses between words)
        word_count = len(text.split())
        if word_count > 1:
            word_rate = word_count / duration
            # Create brief pauses between words
            word_boundaries = np.sin(2 * np.pi * word_rate * t + np.pi/4)
            word_gates = np.where(word_boundaries < -0.8, 0.3, 1.0)  # Brief pauses
            envelope *= word_gates
        
        # Add breath-like variations
        breath_rate = 0.5  # Breathing-like variations
        breath_mod = 1 + 0.1 * np.sin(2 * np.pi * breath_rate * t)
        envelope *= breath_mod
        
        # Character-specific envelope characteristics
        speed = voice_config.get("speed", 1.0)
        if speed > 1.1:  # Fast talkers (like Jinx)
            # More staccato, energetic envelope
            energy_bursts = 1 + 0.2 * (np.random.rand(len(t)) > 0.7).astype(float)
            envelope *= energy_bursts
        elif speed < 0.95:  # Slow, deliberate speakers (like Moses)
            # Smoother, more sustained envelope
            envelope = np.power(envelope, 0.7)  # Gentler amplitude changes
        
        # Ensure envelope doesn't go below minimum level
        envelope = np.maximum(envelope, 0.1)
        
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
        
        logger.debug(f"Generated WAV file: {file_size + 8} bytes total, {data_size} bytes audio data")
        
        # Get bytes and encode to base64
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return f"data:audio/wav;base64,{audio_base64}"
        
    def get_character_voice_info(self, character_id: str) -> dict:
        """Get voice configuration for character"""
        return self.character_voice_configs.get(character_id, {})