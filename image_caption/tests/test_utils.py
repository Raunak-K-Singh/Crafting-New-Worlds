# tests/test_utils.py

import unittest
import os
from src.utils import create_directory

class TestUtils(unittest.TestCase):

    def test_create_directory(self):
        """Test creating a directory."""
        test_dir = 'test_directory'
        
        # Ensure the directory does not exist before the test
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

        create_directory(test_dir)

        # Check if the directory was created
        self.assertTrue(os.path.exists(test_dir))

        # Clean up after test
        os.rmdir(test_dir)

if __name__ == '__main__':
    unittest.main()
