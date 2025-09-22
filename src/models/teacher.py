"""
Teacher model wrapper for PhoWhisper-base.
Provides feature extraction and output generation for knowledge distillation.
"""

import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PhoWhisperTeacher(nn.Module):
    """
    Teacher model wrapper for vinai/PhoWhisper-base.
    Extracts intermediate features and outputs for knowledge distillation.
    """
    
    def __init__(self, model_name: str = "vinai/PhoWhisper-base", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load the pre-trained PhoWhisper model and processor
        logger.info(f"Loading {model_name} model...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Freeze the teacher model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Store intermediate features
        self.intermediate_features = {}
        self.attention_weights = {}
        
        # Register hooks to extract intermediate features
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.intermediate_features[name] = output[0]
                    # Store attention weights if available
                    if len(output) > 1 and hasattr(output[1], 'shape'):
                        self.attention_weights[name] = output[1]
                else:
                    self.intermediate_features[name] = output
            return hook
        
        # Register hooks for encoder layers
        for i, layer in enumerate(self.model.model.encoder.layers):
            layer.register_forward_hook(get_activation(f'encoder_layer_{i}'))
            
        # Register hooks for decoder layers  
        for i, layer in enumerate(self.model.model.decoder.layers):
            layer.register_forward_hook(get_activation(f'decoder_layer_{i}'))
            
        # Register hook for encoder output
        self.model.model.encoder.register_forward_hook(get_activation('encoder_output'))
        
    def preprocess_audio(self, audio_arrays: List[torch.Tensor], sampling_rate: int = 16000) -> Dict:
        """
        Preprocess audio for the teacher model.
        
        Args:
            audio_arrays: List of audio tensors
            sampling_rate: Audio sampling rate
            
        Returns:
            Processed inputs for the model
        """
        # Convert tensors to numpy arrays for the processor
        audio_numpy = [audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio 
                      for audio in audio_arrays]
        
        # Process the audio
        inputs = self.processor(
            audio_numpy,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
        
    def extract_features(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from the teacher model without generating text.
        
        Args:
            input_features: Input mel-spectrogram features
            
        Returns:
            Dictionary containing extracted features and attention weights
        """
        self.intermediate_features.clear()
        self.attention_weights.clear()
        
        with torch.no_grad():
            # Encode the input features
            encoder_outputs = self.model.model.encoder(input_features)
            
            # Get decoder embeddings (we need decoder start token)
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]] * input_features.shape[0]).to(self.device)
            
            # Get embeddings and positional encodings
            decoder_inputs_embeds = self.model.model.decoder.embed_tokens(decoder_input_ids)
            decoder_inputs_embeds = decoder_inputs_embeds + self.model.model.decoder.embed_positions(decoder_input_ids)
            
        return {
            'encoder_features': self.intermediate_features,
            'encoder_output': encoder_outputs.last_hidden_state,
            'attention_weights': self.attention_weights,
            'decoder_embeds': decoder_inputs_embeds
        }
    
    def generate_soft_targets(self, input_features: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
        """
        Generate soft targets (logits) for knowledge distillation.
        
        Args:
            input_features: Input mel-spectrogram features
            temperature: Temperature for softening the distribution
            
        Returns:
            Soft target logits
        """
        with torch.no_grad():
            batch_size = input_features.size(0)
            
            # Create dummy decoder input IDs (start with BOS token)
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.processor.tokenizer.bos_token_id or 50256, 
                dtype=torch.long,
                device=input_features.device
            )
            
            # Generate logits
            outputs = self.model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids
            )
            soft_targets = outputs.logits / temperature
            
        return soft_targets
    
    def forward(self, input_features: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the teacher model.
        
        Args:
            input_features: Input mel-spectrogram features
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing outputs and features
        """
        if return_features:
            features = self.extract_features(input_features)
            soft_targets = self.generate_soft_targets(input_features)
            
            return {
                'soft_targets': soft_targets,
                'encoder_features': features['encoder_features'],
                'encoder_output': features['encoder_output'],
                'attention_weights': features['attention_weights'],
                'decoder_embeds': features['decoder_embeds']
            }
        else:
            with torch.no_grad():
                # Create dummy decoder input IDs for inference
                batch_size = input_features.size(0)
                decoder_input_ids = torch.full(
                    (batch_size, 1), 
                    self.processor.tokenizer.bos_token_id or 50256, 
                    dtype=torch.long,
                    device=input_features.device
                )
                
                outputs = self.model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids
                )
                return {'logits': outputs.logits}
    
    def transcribe(self, audio_arrays: List[torch.Tensor], sampling_rate: int = 16000) -> List[str]:
        """
        Transcribe audio to text using the teacher model.
        
        Args:
            audio_arrays: List of audio tensors
            sampling_rate: Audio sampling rate
            
        Returns:
            List of transcribed texts
        """
        inputs = self.preprocess_audio(audio_arrays, sampling_rate)
        
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
        return transcriptions
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the teacher model."""
        return self.model.config.vocab_size
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the teacher model."""
        return self.model.config.d_model