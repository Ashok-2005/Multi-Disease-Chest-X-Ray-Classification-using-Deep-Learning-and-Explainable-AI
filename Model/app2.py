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
# 🔥 PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Chest X-ray AI",
    page_icon="🫁",
    layout="wide"
)

# ===============================
# 🔥 CUSTOM CSS
# ===============================
st.markdown("""
<style>

html, body, [class*="css"], p, div, span, label {
    font-family: 'Times New Roman', serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Times New Roman', serif !important;
}

.stMarkdown {
    font-family: 'Times New Roman', serif !important;
}

.stApp {
    background-color: white;
}

.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 10px;
    font-family: 'Times New Roman', serif !important;
}

.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 18px;
    margin-bottom: 35px;
    font-family: 'Times New Roman', serif !important;
}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# 🔥 CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "No Finding"
]

NO_FINDING_THRESHOLD = 0.52
DISEASE_THRESHOLD = 0.2

# ===============================
# 🔥 HUMAN FRIENDLY DISEASE INFO
# ===============================
DISEASE_INFO = {

    "Atelectasis":
    "A part of the lung appears collapsed or not fully expanded. This is usually visible in the lower lung regions and may reduce normal airflow.",

    "Cardiomegaly":
    "The heart appears larger than normal in the center chest region. This may indicate enlargement of the heart due to increased workload or heart-related conditions.",

    "Effusion":
    "Fluid appears to be collected around the lungs, usually near the lower chest area. This can make breathing difficult and may occur due to infection or other lung conditions.",

    "Infiltration":
    "Cloudy or abnormal regions are visible inside the lungs. These regions may represent infection, inflammation, or fluid accumulation in lung tissues.",

    "Mass":
    "An unusual dense region is visible inside the lungs. This may represent abnormal tissue growth and usually appears as a localized white area in the chest X-ray.",

    "Nodule":
    "A small rounded spot is visible inside the lung region. Lung nodules are usually small growths that may require further medical evaluation.",

    "Pneumonia":
    "Patch-like white regions are visible in the lungs, which may indicate lung infection. These regions commonly appear in the middle or lower lung areas.",

    "Pneumothorax":
    "Air appears to be trapped around the lungs, causing partial lung collapse. This is commonly seen near the outer edges of the lungs.",

    "Consolidation":
    "Certain lung regions appear denser due to fluid or infection inside the air spaces. This commonly occurs in the middle or lower lung regions.",

    "No Finding":
    "No major abnormality is detected in the chest X-ray image. The lungs and heart appear normal."
}

# ===============================
# 🔥 TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
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

# ===============================
# 🔥 LOAD MODEL
# ===============================
@st.cache_resource
def load_model():

    model = EfficientNetMulti().to(DEVICE)

    model.load_state_dict(
        torch.load("best_model.pth", map_location=DEVICE)
    )

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

        weights = grads.mean(dim=(1, 2))

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

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape

    lung_mask = np.zeros_like(mask)

    lung_mask[
        int(0.2*h):int(0.85*h),
        int(0.2*w):int(0.8*w)
    ] = 255

    mask = cv2.bitwise_and(mask, lung_mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    return cv2.boundingRect(c)

# ===============================
# 🔥 TITLE
# ===============================
st.markdown(
    """
    <div class='big-title'>
    Multi-Disease Chest X-ray Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Deep Learning + Explainable AI using Grad-CAM
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# 🔥 FILE UPLOADER
# ===============================
st.markdown("### Upload Chest X-ray Image")

uploaded_file = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg"]
)

# ===============================
# 🔥 MAIN APP
# ===============================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # ===============================
    # 🔥 TOP LAYOUT
    # ===============================
    left, space, right = st.columns([1, 0.08, 0.95])

    # ===============================
    # 🔥 X-RAY IMAGE
    # ===============================
    with left:

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    # ===============================
    # 🔥 PROBABILITIES
    # ===============================
    with right:

        st.markdown(
            """
            <div style="
                margin-bottom:25px;
            ">
            <h2 style="
                font-family:'Times New Roman', serif;
            ">
            📊 Prediction Probabilities
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, p in enumerate(probs):

            if p >= 0.5:
                emoji = "🔴"

            elif p >= 0.275:
                emoji = "🟠"

            else:
                emoji = "🔵"

            col_a, col_b = st.columns([3,1])

            with col_a:

                st.markdown(
                    f"""
                    <div style="
                        font-size:16px;
                        font-weight:600;
                        margin-top:0px;
                        margin-bottom:0px;
                        font-family:'Times New Roman', serif;
                    ">
                    {emoji} {CLASSES[i]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_b:

                st.markdown(
                    f"""
                    <div style="
                        font-size:16px;
                        font-weight:600;
                        text-align:right;
                        font-family:'Times New Roman', serif;
                    ">
                    {p:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.progress(float(p))

    # ===============================
    # 🔥 DECISION LOGIC
    # ===============================
    no_find_idx = CLASSES.index("No Finding")

    if probs[no_find_idx] > NO_FINDING_THRESHOLD:

        st.success(
            "✅ No major abnormality detected. The chest X-ray appears normal."
        )

    else:

        detected = [
            i for i, p in enumerate(probs)
            if (p > DISEASE_THRESHOLD and i != no_find_idx)
        ]

        if len(detected) == 0:

            st.warning(
                "⚠️ No significant disease detected above the prediction threshold."
            )

        else:

            # ===============================
            # 🔥 DETECTED DISEASES
            # ===============================
            st.markdown("## 🩺 Clinical Findings")

            for i in detected:

                disease = CLASSES[i]

                st.markdown(
                    f"""
                    <div style="
                        padding:14px;
                        border-radius:10px;
                        border:1px solid #d1d5db;
                        margin-bottom:18px;
                        background-color:#fafafa;
                        font-size:17px;
                        line-height:1.7;
                        font-family:'Times New Roman', serif;
                    ">
                    <b>🩺 {disease}</b><br><br>

                    {DISEASE_INFO[disease]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            # ===============================
            # 🔥 GRAD-CAM SECTION
            # ===============================
            st.markdown("## 🔍 Disease Localization using Grad-CAM")

            for i in detected:

                st.markdown(
                f"""
                <div style="
                    text-align:center;
                    margin-top:8px;
                    margin-bottom:8px;
                    font-family:'Times New Roman', serif;
                    font-size:28px;
                    font-weight:700;
                    color:#1f2937;
                ">
                🩺 {CLASSES[i]}
                </div>
                """,
                unsafe_allow_html=True
            )

                cam = cam_gen.generate(x, i)

                cam = cv2.resize(
                    cam,
                    (img_np.shape[1], img_np.shape[0])
                )

                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam),
                    cv2.COLORMAP_JET
                )

                heatmap = cv2.cvtColor(
                    heatmap,
                    cv2.COLOR_BGR2RGB
                )

                overlay = cv2.addWeighted(
                    img_np,
                    0.7,
                    heatmap,
                    0.3,
                    0
                )

                bbox = get_bbox_from_cam(cam)

                if bbox:

                    x1, y1, w, h = bbox

                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x1+w, y1+h),
                        (0,255,0),
                        2
                    )

                left_space, col1, middle_space, col2, right_space = st.columns([0.45, 1, 0.08, 1, 0.45])

                with col1:

                    st.image(
                        img_np,
                        caption="Original Chest X-ray",
                        width=430
                    )

                with col2:

                    st.image(
                        overlay,
                        caption="Grad-CAM Visualization",
                        width=430
                    )
