import os
#import pickle
import pandas as pd
import joblib
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

model = joblib.load(MODEL_PATH)

if not hasattr(model, "predict"):
    raise TypeError(
        f"Loaded object from {MODEL_PATH} is {type(model)} and has no predict(). "
        "Your model.pkl is not a trained sklearn model/pipeline."
    )

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

@app.post("/predict-csv")
def predict_csv():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    df = pd.read_csv(file)

    preds = model.predict(df)
    df["prediction"] = preds

    # Return CSV directly to browser
    import io
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions_output.csv"
    )



@app.get("/downloads/<path:filename>")
def download_file(filename):
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=True)



# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
