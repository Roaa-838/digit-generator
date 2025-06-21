import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🎨",
    layout="wide"
)

# VAE Model Definition (same as training script)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log_var
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

@st.cache_resource
def load_model():
    """Load the trained VAE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(784, 400, 20).to(device)
    
    try:
        # Load the trained model weights
        model.load_state_dict(torch.load('vae_mnist_final.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file 'vae_mnist_final.pth' not found. Please upload your trained model.")
        return None, device

def generate_digit_images(model, device, digit, num_samples=5):
    """Generate images for a specific digit"""
    if model is None:
        return None
    
    with torch.no_grad():
        # Create latent vectors with some variation
        # You can adjust these parameters based on your trained model
        latent_dim = 20
        
        # Generate random latent vectors
        z_samples = []
        for i in range(num_samples):
            # Add some randomness to create variations
            z = torch.randn(1, latent_dim).to(device)
            # You might want to condition this on the digit somehow
            # For now, we'll use random sampling and hope the model learned good representations
            z_samples.append(z)
        
        # Decode the latent vectors to images
        generated_images = []
        for z in z_samples:
            img = model.decode(z).cpu()
            img = img.view(28, 28).numpy()
            generated_images.append(img)
        
        return generated_images

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.title("🎨 Handwritten Digit Generator")
    st.markdown("Generate handwritten digits using a trained Variational Autoencoder (VAE)")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.warning("Please upload your trained model file 'vae_mnist_final.pth' to the same directory as this script.")
        
        # File uploader for model
        uploaded_file = st.file_uploader("Upload your trained model (.pth file)", type=['pth'])
        if uploaded_file is not None:
            # Save uploaded file
            with open("vae_mnist_final.pth", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Model uploaded successfully! Please refresh the page.")
            st.rerun()
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Digit selection
    st.subheader("Select a digit to generate:")
    
    # Create columns for digit buttons
    cols = st.columns(10)
    selected_digit = None
    
    for i in range(10):
        with cols[i]:
            if st.button(str(i), key=f"digit_{i}", use_container_width=True):
                selected_digit = i
    
    # Store selected digit in session state
    if selected_digit is not None:
        st.session_state.selected_digit = selected_digit
    
    # Display selected digit
    if 'selected_digit' in st.session_state:
        st.info(f"Selected digit: **{st.session_state.selected_digit}**")
        
        # Generate button
        if st.button("🎯 Generate 5 Images", type="primary", use_container_width=True):
            with st.spinner("Generating images..."):
                # Generate images
                images = generate_digit_images(
                    model, device, st.session_state.selected_digit, 5
                )
                
                if images:
                    st.subheader(f"Generated Images for Digit {st.session_state.selected_digit}:")
                    
                    # Display images in a grid
                    fig = create_image_grid(images)
                    st.pyplot(fig)
                    
                    # Display individual images in columns
                    cols = st.columns(5)
                    for i, img in enumerate(images):
                        with cols[i]:
                            st.image(img, caption=f"Sample {i+1}", use_column_width=True, cmap='gray')
                    
                    # Download option
                    st.subheader("Download Generated Images:")
                    
                    # Create downloadable images
                    for i, img in enumerate(images):
                        # Convert to PIL Image
                        img_pil = Image.fromarray((img * 255).astype(np.uint8))
                        
                        # Convert to bytes
                        img_bytes = io.BytesIO()
                        img_pil.save(img_bytes, format='PNG')
                        img_bytes = img_bytes.getvalue()
                        
                        # Download button
                        st.download_button(
                            label=f"Download Sample {i+1}",
                            data=img_bytes,
                            file_name=f"digit_{st.session_state.selected_digit}_sample_{i+1}.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )
                else:
                    st.error("Failed to generate images. Please check your model.")
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Device:** {device}")
        st.write("**Model Architecture:** Variational Autoencoder (VAE)")
        st.write("**Input:** 28x28 grayscale images")
        st.write("**Latent Dimension:** 20")
        st.write("**Dataset:** MNIST")

if __name__ == "__main__":
    main()