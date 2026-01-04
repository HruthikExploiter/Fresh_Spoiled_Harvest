import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import requests
from io import BytesIO

MODEL_PATH = os.path.join("artifacts", "fresh_spoiled_optuna_cnn.pth")

class OptunaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.32847146871258026),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

model = OptunaCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.set_page_config(page_title="Fresh_Spoiled_Harvest AI", layout="centered")
st.title("🍎 Fresh_Spoiled_Harvest AI")

st.info("⚠️ This model is trained only on Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, and Tomato. Predictions for other fruits or vegetables may be less accurate.")

st.markdown("""
### How to use
• Upload an image **or**  
• Paste a **direct image URL** (right-click image → *Open image in new tab* → copy link)  
• Click **Predict**
""")

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    img_url = st.text_input("Or paste Image URL")
    predict_btn = st.button("🔍 Predict")

image = None

if predict_btn:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    elif img_url:
        try:
            response = requests.get(img_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("❌ Invalid image URL")
    else:
        st.warning("Please upload an image or paste an image URL")

    if image is not None:
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor)).item()

        with col1:
            if prob > 0.4:
                st.error(f"❌ Spoiled (confidence: {prob:.2f})")
            else:
                st.success(f"✅ Fresh (confidence: {1 - prob:.2f})")

        with col2:
            st.image(image, use_container_width=True)
