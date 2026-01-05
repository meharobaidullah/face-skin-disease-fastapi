from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

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

# Get model input size
_, height, width, channels = model.input_shape

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Skin Disease Classification API")

# --------------------
# Image preprocessing
# --------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((width, height))

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # same normalization as training
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# --------------------
# Prediction endpoint
# --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        predictions = model.predict(img)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = index_to_class[predicted_index]
        confidence = float(predictions[0][predicted_index])

        # All class probabilities
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
