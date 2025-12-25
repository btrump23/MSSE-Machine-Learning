import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__)



# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ------------------------
# Load model
# ------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ------------------------
# Routes
# ------------------------

@app.post("/predict-csv")
def predict_csv():
    return predict()   # reuse the existing predict() function

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_type": str(type(model))
    })

@app.get("/")
def predict_gui():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    df = pd.read_csv(file)

    preds = model.predict(df)
    df["prediction"] = preds

    output_file = "predictions_output.csv"
    output_path = os.path.join(DOWNLOADS_DIR, output_file)
    df.to_csv(output_path, index=False)

    return render_template(
        "index.html",
        download_file=output_file,
        rows=len(df)
    )

@app.get("/downloads/<path:filename>")
def download_file(filename):
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=True)

# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
