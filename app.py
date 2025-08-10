import os
from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_DIR = "models"
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "clf.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "le_labels.pkl"))

IMG_SIZE = (224, 224)
extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

def prepare_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    details = None
    probs = None

    if request.method == "POST":
        f = request.files.get("image")
        if not f or f.filename == "":
            return redirect(request.url)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)

        arr = prepare_image(filepath)
        feat = extractor.predict(arr)
        feat_pca = pca.transform(feat)
        pred_idx = clf.predict(feat_pca)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        prediction = pred_label

        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(feat_pca)[0]
            probs = {le.classes_[i]: float(p[i]) for i in range(len(p))}

        details = f"Predicted: {pred_label}."

    return render_template("index.html", prediction=prediction, details=details, probs=probs)

if __name__ == "__main__":
    app.run(debug=True)
