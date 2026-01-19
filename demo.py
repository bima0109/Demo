import sys
import os
import numpy as np
import cv2
from joblib import load
from skimage import feature
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.transform import resize

COLORS = ["black", "white", "silver"]
MODEL_DIR = "model_output"

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
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
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
        gray,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

def extract_lbp_gray(image, P=8, R=1, n_bins=59):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, (128,128), anti_aliasing=True)
    lbp = local_binary_pattern(
        (gray*255).astype("uint8"),
        P, R, method="uniform"
    )
    hist,_ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0,n_bins+1),
        range=(0,n_bins)
    )
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

def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Gambar tidak bisa dibaca")
    img = random_crop_patch(img)
    return np.hstack([
        extract_color_histogram(img),
        extract_hog_gray(img),
        extract_lbp_gray(img),
        extract_specular_feature(img),
        extract_glcm_contrast(img),
        extract_local_contrast(img)
    ])

def detect_dominant_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = s > 40
    if np.sum(mask) == 0:
        return "white"
    h_mean = np.mean(h[mask])
    v_mean = np.mean(v[mask])
    if v_mean > 200:
        return "white"
    if h_mean < 15 or h_mean > 160:
        return "silver"
    return "black"

def predict_from_features(feat, model, scaler):
    feat = scaler.transform([feat])
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = "Repaint" if pred == 1 else "Non Repaint"
    return label, prob[pred]*100

def main():
    if len(sys.argv) != 2:
        print("Usage: python demo.py namafile.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print("File tidak ditemukan")
        sys.exit(1)

    global_model = load(os.path.join(MODEL_DIR,"model.joblib"))
    scaler = load(os.path.join(MODEL_DIR,"scaler.joblib"))

    models_per_color = {}
    for c in COLORS:
        p = os.path.join(MODEL_DIR,f"svm_{c}.joblib")
        if os.path.exists(p):
            models_per_color[c] = load(p)

    img = cv2.imread(img_path)
    detected_color = detect_dominant_color(img)
    feat = extract_features(img_path)

    label_g, conf_g = predict_from_features(feat, global_model, scaler)

    print(f"\n[DEMO] {os.path.basename(img_path)}")
    print(f"GLOBAL   → {label_g} ({conf_g:.2f}%)")

    if detected_color in models_per_color:
        label_c, conf_c = predict_from_features(
            feat, models_per_color[detected_color], scaler
        )
        print(f"PerColor → {label_c} ({conf_c:.2f}%)")

if __name__ == "__main__":
    main()
