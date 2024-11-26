from dataclasses import dataclass, field
from typing import List, Dict
import torch

@dataclass
class ModelConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: torch.dtype = torch.float16  # Default to half precision
    image_size: int = 512  # Increased for better detail
    batch_size: int = 4  # Increased for better throughput
    max_caption_length: int = 200
    beam_size: int = 12  # Increased for better caption diversity
    temperature: float = 0.8
    top_k: int = 100
    top_p: float = 0.95
    repetition_penalty: float = 1.2  # Added to prevent repetitive captions
    
    model_configs: Dict[str, List[str]] = field(default_factory=lambda: {
        "image_encoder": [
            "google/vit-large-patch16-512",
            "google/vit-huge-patch14-512",
            "facebook/deit-v2-large-384"
        ],
        "clip": [
            "openai/clip-vit-large-patch14-336",
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ],
        "blip": [
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-flan-t5-xl"
        ],
        "caption_decoder": [
            "facebook/opt-13b",
            "bigscience/bloom-7b1",
            "EleutherAI/gpt-neox-20b"
        ]
    })
    
    learning_rate: float = 5e-6
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    mixed_precision_training: bool = True
    
    confidence_threshold: float = 0.85
    max_entities_per_image: int = 30
    max_relations_per_image: int = 50
    min_entity_similarity: float = 0.75
    
    enable_emotion_detection: bool = True
    enable_style_transfer: bool = True

    caption_styles: List[str] = field(default_factory=lambda: [
        "descriptive", "narrative", "technical", "poetic", "humorous"
    ])
