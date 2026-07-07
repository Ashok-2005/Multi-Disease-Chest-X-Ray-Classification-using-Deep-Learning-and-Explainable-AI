# ===============================
# 🔥 IMPORTS
# ===============================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ===============================
# 🔥 CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","No Finding"
]

# ===============================
# 🔥 MODEL
# ===============================
class EfficientNetXAI(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, len(CLASSES))

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out = self.classifier(pooled)
        return out, feat

model = EfficientNetXAI().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# 🔥 Fix randomness
torch.manual_seed(0)
np.random.seed(0)

# ===============================
# 🔥 TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# 🔥 X-RAY CHECK
# ===============================
def is_chest_xray(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    if len(img_np.shape) == 3:
        diff = np.abs(img_np[:,:,0] - img_np[:,:,1]).mean() + \
               np.abs(img_np[:,:,1] - img_np[:,:,2]).mean()
        if diff > 40:
            return False

    if gray.std() < 10:
        return False

    edges = cv2.Canny(gray, 30, 120)
    if edges.mean() < 2:
        return False

    return True

# ===============================
# ===============================
# 🔥 GRADCAM++ (CORRECTED)
# ===============================
class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # ✅ Use LAST convolution layer (IMPORTANT FIX)
        self.target_layer = self.model.features[-1]

        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()

        logits, _ = self.model(x)

        # 🔥 Strong class isolation (IMPORTANT)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1

        score = torch.sum(one_hot * logits)

        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        # 🔥 Normalize gradients (KEY FIX)
        gradients = gradients / (torch.sqrt(torch.mean(gradients**2)) + 1e-8)

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(x.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam
# ===============================
# 🔥 LUNG MASK
# ===============================
def lung_mask(cam, orig_img):
    img = np.array(orig_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((9,9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(thresh)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for c in contours:
            cv2.drawContours(mask, [c], -1, 255, -1)

    mask = cv2.erode(mask, np.ones((15,15), np.uint8))
    mask = cv2.resize(mask, (cam.shape[1], cam.shape[0]))

    return cam * (mask / 255.0)

# ===============================
# 🔥 BOUNDING BOX
# ===============================
def get_bbox(cam, orig_img):
    cam_uint8 = np.uint8(255 * cam)
    _, thresh = cv2.threshold(cam_uint8, 80, 255, cv2.THRESH_BINARY)

    img = np.array(orig_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, lung = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((9,9), np.uint8)
    lung = cv2.morphologyEx(lung, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(lung)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for c in contours:
            cv2.drawContours(mask, [c], -1, 255, -1)

    mask = cv2.erode(mask, np.ones((25,25), np.uint8))
    thresh = cv2.bitwise_and(thresh, mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return cv2.boundingRect(max(contours, key=cv2.contourArea))

# ===============================
# 🔥 SHOW CAM (IDENTICAL OUTPUT)
# ===============================
def show_cam(img, cam, title=""):
    img_np = np.array(img)

    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    cam = lung_mask(cam, img)

    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    bbox = get_bbox(cam, img)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].imshow(img_np, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title(title)
    ax[1].axis("off")

    st.pyplot(fig)

# ===============================
# 🔥 STREAMLIT UI (ONLY CHANGE)
# ===============================
st.set_page_config(layout="wide")

st.title("MULTI-DISEASE CHEST X-RAY CLASSIFICATION")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    orig_img = Image.open(uploaded).convert("RGB")
    img_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)

    if not is_chest_xray(orig_img):
        st.error("❌ Error: Not a Chest X-ray image.")
    else:
        with torch.no_grad():
            logits, _ = model(img_tensor)

        probs = torch.sigmoid(logits)[0].cpu().numpy()
        cam_gen = GradCAMPlusPlus(model)

        st.markdown("<h3>Predicted Probabilities</h3>", unsafe_allow_html=True)
        for i, p in enumerate(probs):
            st.text(f"{CLASSES[i]:15}: {p:.4f}")

        st.markdown("<h3>Possible Findings</h3>", unsafe_allow_html=True)

        detected = []
        no_idx = CLASSES.index("No Finding")

        thresholds = {
            "Atelectasis": 0.205,
            "Cardiomegaly": 0.10,
            "Effusion": 0.20,
            "Infiltration": 0.21,
            "Mass": 0.15,
            "Nodule": 0.18,
            "Pneumonia": 0.07,
            "Pneumothorax": 0.15,
            "Consolidation": 0.10,
            "No Finding": 0.54
        }

        if probs[no_idx] > thresholds["No Finding"]:
            st.success("No disease detected. The chest X-ray appears normal.")
            detected.append((no_idx, probs[no_idx]))

        else:
            for i, prob in enumerate(probs):

                disease_name = CLASSES[i]

                if disease_name == "No Finding":
                    continue

                if prob > thresholds[disease_name]:
                    st.write(f"Possible sign of {disease_name} detected.")
                    detected.append((i, prob))

        for idx, prob in detected:
            if CLASSES[idx] == "No Finding":
                continue

            cam = cam_gen.generate(img_tensor, idx)
            show_cam(orig_img, cam, f"{CLASSES[idx]}")
