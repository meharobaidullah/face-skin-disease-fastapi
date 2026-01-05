from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from typing import List

# --------------------
# Load model once
# --------------------
model = load_model("model.hdf5")

# --------------------
# Class mappings
# --------------------
class_indices = {
    'Acne': 0,
    'Actinic Keratosis': 1,
    'Basal Cell Carcinoma': 2,
    'Eczemaa': 3,
    'Rosacea': 4
}
index_to_class = {v: k for k, v in class_indices.items()}

# Model input shape
_, height, width, _ = model.input_shape

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Skin Disease Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Preprocess image
# --------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((width, height))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array

# --------------------
# Single image prediction
# --------------------
@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = index_to_class[predicted_index]
        confidence = float(predictions[0][predicted_index])

        probabilities = {
            index_to_class[i]: float(predictions[0][i])
            for i in range(len(predictions[0]))
        }

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# --------------------
# Batch image prediction
# --------------------
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        return JSONResponse(status_code=400, content={"error": "No images uploaded"})

    try:
        images = []

        for file in files:
            image_bytes = await file.read()
            img = preprocess_image(image_bytes)
            images.append(img)

        # Shape: (N, height, width, 3)
        batch = np.stack(images, axis=0)

        predictions = model.predict(batch)  # (N, 5)

        # Average probabilities across images
        avg_probabilities = np.mean(predictions, axis=0)

        predicted_index = int(np.argmax(avg_probabilities))
        predicted_class = index_to_class[predicted_index]
        confidence = float(avg_probabilities[predicted_index])

        probabilities = {
            index_to_class[i]: float(avg_probabilities[i])
            for i in range(len(avg_probabilities))
        }

        return JSONResponse({
            "num_images": len(files),
            "final_prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
