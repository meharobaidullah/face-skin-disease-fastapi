from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import io
from typing import List, Tuple

# --------------------
# Configuration
# --------------------
MODEL_PATH = "my_model.h5"
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

class_indices = {
    'Eczema': 0,
    'Viral Infections': 1,
    'Melanoma': 2,
    'Atopic Dermatitis': 3,
    'Basal Cell Carcinoma': 4,
    'Melanocytic Nevi': 5,
    'Keratosis-like Lesions': 6,
    'Psoriasis & Lichen Planus': 7,
    'Seborrheic Keratoses': 8,
    'Fungal Infections': 9
}
index_to_class = {v: k for k, v in class_indices.items()}
EXPECTED_NUM_CLASSES = len(class_indices)

def load_and_validate_model(model_path: str) -> Tuple[object, int, int]:
    loaded_model = load_model(model_path)

    input_shape = loaded_model.input_shape
    if isinstance(input_shape, list):
        raise ValueError("Model must have a single image input.")
    if len(input_shape) != 4:
        raise ValueError(f"Expected input shape rank 4, got {input_shape}.")

    _, height, width, channels = input_shape
    if channels != 3:
        raise ValueError(f"Expected 3 input channels (RGB), got {channels}.")
    if height is None or width is None:
        raise ValueError(f"Model input height/width must be fixed, got {input_shape}.")

    output_shape = loaded_model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("Model must have a single output tensor.")
    num_classes = output_shape[-1]
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Model output classes ({num_classes}) do not match expected "
            f"API classes ({EXPECTED_NUM_CLASSES})."
        )

    return loaded_model, int(height), int(width)


# --------------------
# Load model once
# --------------------
model, height, width = load_and_validate_model(MODEL_PATH)

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
    # Match training-time preprocessing for ResNet-based models.
    img_array = preprocess_input(np.array(image, dtype=np.float32))
    return img_array


def response_no_cache(content, status_code=200):
    return JSONResponse(content=content, status_code=status_code, headers=NO_CACHE_HEADERS)

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

        return response_no_cache({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return response_no_cache(
            {
                "error": str(e)
            },
            status_code=500,
        )

# --------------------
# Batch image prediction
# --------------------
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        return response_no_cache(
            {"error": "No images uploaded"},
            status_code=400,
        )

    try:
        images = []

        for file in files:
            image_bytes = await file.read()
            img = preprocess_image(image_bytes)
            images.append(img)

        # Shape: (N, height, width, 3)
        batch = np.stack(images, axis=0)

        predictions = model.predict(batch)  # (N, num_classes)

        # Average probabilities across images
        avg_probabilities = np.mean(predictions, axis=0)

        predicted_index = int(np.argmax(avg_probabilities))
        predicted_class = index_to_class[predicted_index]
        confidence = float(avg_probabilities[predicted_index])

        probabilities = {
            index_to_class[i]: float(avg_probabilities[i])
            for i in range(len(avg_probabilities))
        }

        return response_no_cache({
            "num_images": len(files),
            "final_prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return response_no_cache(
            {
                "error": str(e)
            },
            status_code=500,
        )
