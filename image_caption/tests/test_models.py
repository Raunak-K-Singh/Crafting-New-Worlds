import unittest
import torch
from src.models import EnhancedImageEncoder  # Adjust the import based on your structure
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

        # Check shapes of outputs (adjust based on your actual model output shapes)
        self.assertEqual(output['features'].shape, (self.config.batch_size, 2048))  # Example shape
        self.assertEqual(output['style_features'].shape, (self.config.batch_size, 256))  # Example shape
        self.assertEqual(output['emotions'].shape, (self.config.batch_size, 5))  # Example shape for emotion classes

if __name__ == '__main__':
    unittest.main()
