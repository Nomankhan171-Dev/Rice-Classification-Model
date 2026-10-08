
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rice_classifier.pkl"

def extract_features(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edges = float(gx + gy)
    center = gray[56:168,56:168].mean()
    border = np.concatenate([
        gray[:30,:].ravel(), gray[-30:,:].ravel(),
        gray[:, :30].ravel(), gray[:, -30:].ravel()
    ]).mean()
    center_delta = float(center - border)
    return np.array([
        *means.tolist(),
        *stds.tolist(),
        brightness, contrast, saturation, edges, center_delta
    ], dtype=np.float32)

X, y = [], []

if not DATASET_DIR.exists():
    raise SystemExit("Create dataset/class-name folders first.")

for class_dir in sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()]):
    for file in class_dir.iterdir():
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            X.append(extract_features(Image.open(file)))
            y.append(class_dir.name)
        except Exception as exc:
            print("Skipping", file, exc)

X = np.asarray(X)
y = np.asarray(y)

if len(X) < 20 or len(set(y)) < 2:
    raise SystemExit("Not enough training data. Add images for at least 2 classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Saved model to:", MODEL_PATH)
