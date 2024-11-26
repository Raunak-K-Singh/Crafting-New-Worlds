# tests/test_models.py

import unittest
import torch
from src.models import EnhancedImageEncoder  # Adjust based on your structure
from src.config import ModelConfig

class TestEnhancedImageEncoder(unittest.TestCase):

    def setUp(self):
        """Set up the model configuration and encoder for testing."""
        self.config = ModelConfig()
        self.model = EnhancedImageEncoder(self.config)

    def test_forward_pass(self):
        """Test the forward pass of the EnhancedImageEncoder."""
        sample_images = torch.randn(self.config.batch_size, 3, self.config.image_size, self.config.image_size)
        output = self.model(sample_images)
        
        # Check if output contains expected keys
        self.assertIn('features', output)
        self.assertIn('style_features', output)
        self.assertIn('emotions', output)

if __name__ == '__main__':
    unittest.main()
