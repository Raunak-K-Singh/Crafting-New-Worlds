import os

def create_directory(dir_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def load_config(file_path):
    """Load configuration from a JSON file."""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def save_model(model, path):
    """Save a PyTorch model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load a PyTorch model from the specified path."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
