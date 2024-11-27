import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.models import EnhancedImageEncoder  # Adjust based on your structure
from src.config import ModelConfig  # Ensure this points to your configuration

# Load your model (make sure to adjust the path and loading mechanism)
@st.cache(allow_output_mutation=True)
def load_model():
    config = ModelConfig()  # Load your configuration
    model = EnhancedImageEncoder(config)  # Initialize the model
    model.load_state_dict(torch.load("path/to/your/model.pth"))  # Load the trained model weights
    model.eval()  # Set the model to evaluation mode
    return model

# Define image transformations
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Main function to run the app
def main():
    st.title("Image Captioning with Deep Learning")
    
    st.write("Upload an image to generate a caption.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Generate Caption"):
            model = load_model()  # Load the model
            
            # Preprocess the image for the model
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)  # Forward pass through the model
            
            # Assuming output is a tensor containing captions or indices of words
            caption = "Generated caption placeholder"  # Replace with actual caption processing logic
            
            st.write(caption)

if __name__ == "__main__":
    main()
