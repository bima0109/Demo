import os
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from joblib import load
from skimage import feature
import plotly.graph_objects as go
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.transform import resize

COLORS = ["black", "white", "silver"]
MODEL_DIR = "model_output"

@st.cache_resource
def load_models():
    global_model = load(os.path.join(MODEL_DIR, "model.joblib"))
    scaler = load(os.path.join(MODEL_DIR, "scaler.joblib"))
    per_color = {}
    for c in COLORS:
        p = os.path.join(MODEL_DIR, f"svm_{c}.joblib")
        if os.path.exists(p):
            per_color[c] = load(p)
    return global_model, scaler, per_color

def extract_specular_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 230)
    ratio = bright_pixels / gray.size
    max_intensity = gray.max() / 255.0
    return np.array([ratio, max_intensity])

def extract_glcm_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, (64,64), anti_aliasing=True)
    gray = (gray * 255).astype("uint8")
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'energy')[0,0]
    ])

def extract_local_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.array([gray.std() / 255.0])

def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, (128,128), anti_aliasing=True)
    gray = (gray * 255).astype("uint8")
    return feature.hog(
        gray, orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

def extract_lbp_gray(image, P=8, R=1, n_bins=59):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, (128,128), anti_aliasing=True)
    lbp = local_binary_pattern((gray*255).astype("uint8"), P, R, method="uniform")
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,n_bins+1))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def random_crop_patch(img, patch_size=64):
    h,w,_ = img.shape
    if h < patch_size or w < patch_size:
        return cv2.resize(img,(patch_size,patch_size))
    y = np.random.randint(0,h-patch_size)
    x = np.random.randint(0,w-patch_size)
    return img[y:y+patch_size, x:x+patch_size]

def extract_features(img):
    img = random_crop_patch(img)
    return np.hstack([
        extract_color_histogram(img),
        extract_hog_gray(img),
        extract_lbp_gray(img),
        extract_specular_feature(img),
        extract_glcm_contrast(img),
        extract_local_contrast(img)
    ])

def extract_features_multi_crop(img, n_crop=10):
    feats = []
    for _ in range(n_crop):
        feats.append(extract_features(img))
    return feats


def detect_dominant_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = s > 40
    if np.sum(mask) == 0:
        return "white"
    if np.mean(v[mask]) > 200:
        return "white"
    if np.mean(h[mask]) < 15 or np.mean(h[mask]) > 160:
        return "silver"
    return "black"

def predict(feat, model, scaler):
    feat = scaler.transform([feat])
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    return ("Repaint" if pred==1 else "Non Repaint"), prob[pred]*100

def predict_average(feat, global_model, per_color_models, scaler):
    probs = []

    feat_scaled = scaler.transform([feat])

    prob_global = global_model.predict_proba(feat_scaled)[0][1]
    probs.append(prob_global)

    for model in per_color_models.values():
        prob_c = model.predict_proba(feat_scaled)[0][1]
        probs.append(prob_c)

    avg_prob = np.mean(probs)

    label = "Repaint" if avg_prob >= 0.5 else "Non Repaint"
    return label, avg_prob * 100


def predict_multi_crop(feats, global_model, per_color_models, scaler):
    crop_probs = []

    for feat in feats:
        feat_scaled = scaler.transform([feat])

        probs = []
        probs.append(global_model.predict_proba(feat_scaled)[0][1])

        for model in per_color_models.values():
            probs.append(model.predict_proba(feat_scaled)[0][1])

        crop_probs.append(np.mean(probs))

    avg_prob = np.mean(crop_probs)
    label = "Repaint" if avg_prob >= 0.5 else "Non Repaint"

    return label, avg_prob * 100, crop_probs

def confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if confidence >= 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#2ecc71"},
                {'range': [50, 100], 'color': "#e74c3c"}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_multi_crop_voting(crop_probs):
    df = pd.DataFrame({
        "Patch": np.arange(1, len(crop_probs)+1),
        "Probabilitas Repaint (%)": np.array(crop_probs) * 100
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Patch"],
        y=df["Probabilitas Repaint (%)"],
        marker_color="#3498db"
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Multi-Crop Voting Probability",
        xaxis_title="Patch ke-",
        yaxis_title="Probabilitas Repaint (%)",
        height=350
    )

    return fig


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Deteksi Cat Mobil", layout="centered")
st.markdown(
    "<h1 style='text-align:center;'>Sistem Deteksi Keaslian Cat Mobil</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px;'>Analisis Citra Permukaan Cat Mobil Berbasis Machine Learning</p>",
    unsafe_allow_html=True
)


# uploaded = st.file_uploader("Upload gambar mobil", type=["jpg","png","jpeg"])

uploaded = st.file_uploader(
    "Unggah Citra Kendaraan",
    type=["jpg", "png", "jpeg"]
)


if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Citra Input")

    global_model, scaler, models_per_color = load_models()

    with st.spinner("Menganalisis citra..."):
        feats = extract_features_multi_crop(img, n_crop=10)
        label_avg, conf_avg, crop_probs = predict_multi_crop(
            feats,
            global_model,
            models_per_color,
            scaler
        )

    st.markdown("## 📊 Hasil Prediksi Akhir")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keputusan Sistem", label_avg)
    with col2:
        st.metric("Tingkat Keyakinan", f"{conf_avg:.2f}%")
    

    st.markdown("## 📈 Analisis Keputusan Model")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Multi-Crop Voting")
        st.plotly_chart(
            plot_multi_crop_voting(crop_probs),
            use_container_width=True
        )

    with right_col:
        st.markdown("### Confidence Gauge")
        st.plotly_chart(
            confidence_gauge(conf_avg),
            use_container_width=True
        )
 


# ================= THEME SWITCH =================
theme = st.toggle("🌗 Dark Mode", value=True)

if theme:
    bg_color = "rgba(15, 15, 15, 0.85)"
    text_color = "#ffffff"
    card_color = "rgba(30, 30, 30, 0.85)"
else:
    bg_color = "rgba(255, 255, 255, 0.85)"
    text_color = "#000000"
    card_color = "rgba(245, 245, 245, 0.85)"

# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url("https://png.pngtree.com/png-clipart/20230508/original/pngtree-sport-car-silhouette-png-image_9152154.png");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }}

#     h1, h2, h3, h4, h5, h6, p, label {{
#         color: {text_color} !important;
#     }}

#     .block-container {{
#         background-color: {bg_color};
#         padding: 2rem;
#         border-radius: 16px;
#         box-shadow: 0 8px 30px rgba(0,0,0,0.35);
#     }}

#     div[data-testid="stMetric"] {{
#         background-color: {card_color};
#         padding: 1.2rem;
#         border-radius: 14px;
#         box-shadow: 0 4px 20px rgba(0,0,0,0.25);
#     }}

#     div[data-testid="stFileUploader"] {{
#         background-color: {card_color};
#         padding: 1rem;
#         border-radius: 12px;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://static.vecteezy.com/system/resources/previews/003/586/513/non_2x/sports-car-silhouette-vector.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: {text_color};
    }}

    body, div, span, p, label, h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}

    .block-container {{
        background-color: {bg_color};
        padding: 1.5rem 2rem 2rem 2rem;
        margin-top: 3rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }}

    div[data-testid="stMetric"] {{
        background-color: {card_color};
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }}

    div[data-testid="stFileUploader"] {{
        background-color: {card_color};
        padding: 1rem;
        border-radius: 12px;
    }}
    
    
    </style>
    """,
    unsafe_allow_html=True
)

