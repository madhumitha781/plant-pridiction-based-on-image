import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

DATA_DIR = "data/train"
IMG_SIZE = (224, 224)
BATCH = 32
PCA_COMPONENTS = 128
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_images_and_labels(data_dir):
    X = []
    y = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                arr = img_to_array(img)
                X.append(arr)
                y.append(cls)
            except Exception as e:
                print("skip", path, e)
    X = np.array(X, dtype="float32")
    return X, np.array(y)

def extract_features(X, batch_size=32):
    extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                             input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    X_p = preprocess_input(X)
    features = extractor.predict(X_p, batch_size=batch_size, verbose=1)
    return features

def main():
    print("Loading images...")
    X_imgs, y_labels = load_images_and_labels(DATA_DIR)
    print("Loaded:", X_imgs.shape, "labels:", y_labels.shape)

    print("Extracting CNN features (MobileNetV2)...")
    feats = extract_features(X_imgs, batch_size=BATCH)
    print("Feature shape:", feats.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(feats, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    print(f"Fitting PCA ({PCA_COMPONENTS} components)...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Training classifier (SVC)...")
    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "le_labels.pkl"))
    print("Saved models to", MODEL_DIR)

if __name__ == "__main__":
    main()
