import torch
from config import ModelConfig
from models import EnhancedImageEncoder

def main():
   # Initialize model configuration.
   config=ModelConfig()

   # Create an instance of the EnhancedImageEncoder.
   image_encoder=EnhancedImageEncoder(config)

   # Create a sample input tensor representing a batch of images (batch size of 4).
   sample_images=torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(config.device)

   print("Sample images created.")

   # Forward pass through the enhanced image encoder.
   output=image_encoder(sample_images)

   print("Output from Enhanced Image Encoder:")
   print(f"Fused Features Shape: {output['features'].shape}")
   print(f"Style Features Shape: {output['style_features'].shape if output['style_features'] is not None else 'N/A'}")
   print(f"Emotions Shape: {output['emotions'].shape if output['emotions'] is not None else 'N/A'}")

if __name__ == "__main__":
   main()
