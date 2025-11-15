import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Leaf Classifier",
    page_icon="🍃",
    layout="centered"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (update based on your dataset)
CLASS_NAMES = ['Smooth', 'Serrated']  # Adjust if different

@st.cache_resource
def load_model():
    """Load the trained ConvNeXt model."""
    # Create model architecture
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASS_NAMES))
    
    # Load trained weights
    model.load_state_dict(torch.load('custom_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model

def get_transforms():
    """Get image preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict(image, model, transform):
    """Make prediction on a single image."""
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# Main app
def main():
    st.title("🍃 Leaf Classifier")
    st.markdown("### Classify leaves as **Smooth** or **Serrated**")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
        transform = get_transforms()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a leaf to classify"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        with st.spinner("Analyzing..."):
            predicted_class, confidence, all_probs = predict(image, model, transform)
        
        with col2:
            st.markdown("### Prediction Results")
            st.markdown(f"**Class:** {CLASS_NAMES[predicted_class]}")
            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
            
            # Display probabilities for all classes
            st.markdown("### Class Probabilities")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = all_probs[i] * 100
                st.progress(float(all_probs[i]))
                st.text(f"{class_name}: {prob:.2f}%")
        
        # Additional info
        st.markdown("---")
        st.info(f"✨ Model: ConvNeXt Tiny | Device: {device}")
    
    else:
        st.info("👆 Please upload a leaf image to get started")
        
        # Optional: Add example images or instructions
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Click on 'Browse files' above
            2. Select a leaf image from your device
            3. Wait for the model to analyze the image
            4. View the classification results!
            
            **Note:** This model classifies leaves into two categories:
            - **Smooth**: Leaves with smooth edges
            - **Serrated**: Leaves with jagged/toothed edges
            """)

if __name__ == "__main__":
    main()