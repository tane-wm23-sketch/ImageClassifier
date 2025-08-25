import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# ===== Model Preparation =====
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    model.load_state_dict(torch.load("final_best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ===== Image Transform (must match training preprocessing) =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ===== Streamlit UI =====
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
st.title("CIFAR-10 Image Classifier")

# Initialize history in session_state
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, 1).item()

    # Save to history
    st.session_state.history.append({
        "filename": uploaded_file.name,
        "prediction": classes[predicted_class],
        "probs": probs.numpy()[0],
        "image": img
    })

# Show history
if st.session_state.history:
    st.subheader("Prediction History")
    for item in st.session_state.history[::-1]:  # show latest first
        st.image(item["image"], caption=f"File: {item['filename']}", use_container_width=True)
        st.write(f"**Class: {item['prediction']}**")
        probs_df = pd.DataFrame({
            "Class": classes,
            "Probability": item["probs"]
        })
        st.bar_chart(probs_df.set_index("Class"))