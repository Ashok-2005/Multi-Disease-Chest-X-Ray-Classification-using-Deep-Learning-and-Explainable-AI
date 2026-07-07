# ===============================
# 🔥 IMPORTS
# ===============================
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# ===============================
# 🔥 CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","No Finding"
]

NO_FINDING_THRESHOLD = 0.5
DISEASE_THRESHOLD = 0.275

# ===============================
# 🔥 TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===============================
# 🔥 MODEL
# ===============================
class EfficientNetMulti(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, len(CLASSES))

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(pooled)

@st.cache_resource
def load_model():
    model = EfficientNetMulti().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ===============================
# 🔥 GRAD-CAM
# ===============================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        self.target_layer = model.features[-1]
        self.target_layer.register_forward_hook(self.fwd)
        self.target_layer.register_full_backward_hook(self.bwd)

    def fwd(self, m, i, o):
        self.activations = o

    def bwd(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1

        score = torch.sum(one_hot * logits)
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1,2))

        cam = torch.zeros(acts.shape[1:], device=x.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

cam_gen = GradCAM(model)

# ===============================
# 🔥 BBOX FUNCTION
# ===============================
def get_bbox_from_cam(cam):
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    thresh_val = np.percentile(cam, 80)
    mask = cam > thresh_val
    mask = np.uint8(mask * 255)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    lung_mask = np.zeros_like(mask)
    lung_mask[int(0.2*h):int(0.85*h), int(0.2*w):int(0.8*w)] = 255

    mask = cv2.bitwise_and(mask, lung_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)

# ===============================
# 🔥 STREAMLIT UI
# ===============================
st.title("🫁 Chest X-ray Disease Detection")

uploaded_file = st.file_uploader("Upload X-ray", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # -------------------------------
    # 📊 SHOW PROBABILITIES FIRST
    # -------------------------------
    st.subheader("📊 Prediction Probabilities")

    for i, p in enumerate(probs):
        st.write(f"{CLASSES[i]}: {p:.4f}")

    no_find_idx = CLASSES.index("No Finding")

    # -------------------------------
    # 🔥 DECISION LOGIC
    # -------------------------------
    if probs[no_find_idx] > NO_FINDING_THRESHOLD:
        st.success("✅ NORMAL (No Finding)")

    else:
        detected = [
            i for i, p in enumerate(probs)
            if (p > DISEASE_THRESHOLD and i != no_find_idx)
        ]

        if len(detected) == 0:
            st.warning("⚠️ Likely NORMAL (no disease above threshold)")
        else:
            # -------------------------------
            # 🦠 SHOW DETECTED DISEASES
            # -------------------------------
            st.subheader("🦠 Detected Diseases")

            for i in detected:
                st.write(f"👉 {CLASSES[i]}")

            st.divider()

            # -------------------------------
            # 🔥 SHOW VISUALIZATION AFTER
            # -------------------------------
            for i in detected:
                st.markdown(f"### 🔍 {CLASSES[i]}")

                cam = cam_gen.generate(x, i)
                cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam), cv2.COLORMAP_JET
                )
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

                bbox = get_bbox_from_cam(cam)
                if bbox:
                    x1, y1, w, h = bbox
                    cv2.rectangle(overlay,
                                  (x1, y1),
                                  (x1+w, y1+h),
                                  (0,255,0), 2)

                # 🔥 SIDE-BY-SIDE DISPLAY
                col1, col2 = st.columns(2)

                with col1:
                    st.image(img_np, caption="Original X-ray")

                with col2:
                    st.image(overlay, caption="Grad-CAM + Bounding Box")
