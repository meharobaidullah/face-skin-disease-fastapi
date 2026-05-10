from __future__ import annotations

import asyncio
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Disable GPU usage before TensorFlow is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

MODEL_REPO = os.getenv("MODEL_REPO", "REPLACE_ME")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "my_model.h5")
MODEL_CACHE_PATH = Path("/tmp") / MODEL_FILENAME
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
DEFAULT_IMAGE_SIZE = 224

LABELS = [
    "Eczema",
    "Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Keratosis-like Lesions",
    "Psoriasis & Lichen Planus",
    "Seborrheic Keratoses",
    "Fungal Infections",
]

INDEX_TO_LABEL = {index: label for index, label in enumerate(LABELS)}
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

model: tf.keras.Model | None = None
image_height = DEFAULT_IMAGE_SIZE
image_width = DEFAULT_IMAGE_SIZE
model_lock = threading.Lock()


def _raise_http_error(status_code: int, detail: str) -> None:
    raise HTTPException(status_code=status_code, detail=detail)


def _load_model_from_hugging_face() -> Tuple[tf.keras.Model, int, int]:
    """Download the model once and infer the expected image size."""
    if MODEL_REPO == "REPLACE_ME":
        _raise_http_error(
            500,
            "MODEL_REPO is not configured. Set MODEL_REPO to your Hugging Face repo id.",
        )

    downloaded_path = str(MODEL_CACHE_PATH)
    if not MODEL_CACHE_PATH.exists():
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir="/tmp",
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN"),
        )

    loaded_model = tf.keras.models.load_model(downloaded_path)

    input_shape = loaded_model.input_shape
    if isinstance(input_shape, list):
        _raise_http_error(500, "The model must expose a single image input tensor.")
    if len(input_shape) != 4:
        _raise_http_error(500, f"Expected a rank-4 image input, got {input_shape}.")

    _, inferred_height, inferred_width, channels = input_shape
    if channels not in (3, None):
        _raise_http_error(500, f"Expected RGB input with 3 channels, got {input_shape}.")

    height = int(inferred_height) if inferred_height is not None else DEFAULT_IMAGE_SIZE
    width = int(inferred_width) if inferred_width is not None else DEFAULT_IMAGE_SIZE

    output_shape = loaded_model.output_shape
    if isinstance(output_shape, list):
        _raise_http_error(500, "The model must expose a single output tensor.")

    if output_shape[-1] != len(LABELS):
        _raise_http_error(
            500,
            f"Model output size {output_shape[-1]} does not match the required label set of {len(LABELS)} classes.",
        )

    return loaded_model, height, width


def load_model_once() -> None:
    global model, image_height, image_width

    if model is not None:
        return

    with model_lock:
        if model is not None:
            return

        loaded_model, height, width = _load_model_from_hugging_face()
        model = loaded_model
        image_height = height
        image_width = width


def _no_cache_response(content: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=content, status_code=status_code, headers=NO_CACHE_HEADERS)


def _validate_image_bytes(file_bytes: bytes, file_name: str | None, content_type: str | None) -> bytes:
    if not file_bytes:
        _raise_http_error(400, "Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        _raise_http_error(413, "File size exceeds the 5MB limit.")

    if content_type and not content_type.startswith("image/"):
        _raise_http_error(415, f"Unsupported media type for {file_name or 'uploaded file'}.")

    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        _raise_http_error(415, f"Uploaded file {file_name or 'file'} is not a valid image.")

    return file_bytes


def _preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize((image_width, image_height))
            image_array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        _raise_http_error(415, f"Invalid image data: {exc}")

    return image_array


def _predict_batch_numpy(batch: np.ndarray) -> np.ndarray:
    if model is None:
        _raise_http_error(500, "Model is not loaded.")

    predictions = model.predict(batch, verbose=0)
    return np.asarray(predictions, dtype=np.float32)


def _format_probabilities(probabilities: np.ndarray) -> Dict[str, float]:
    return {
        INDEX_TO_LABEL[index]: float(probabilities[index])
        for index in range(len(LABELS))
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_once()
    yield


app = FastAPI(title="Skin Disease Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://face-skin-disease-frontend.devfuze.workers.dev",
        "http://localhost.*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)) -> JSONResponse:
    file_bytes = await file.read()
    _validate_image_bytes(file_bytes, file.filename, file.content_type)

    image_array = _preprocess_image_bytes(file_bytes)
    batch = np.expand_dims(image_array, axis=0)

    predictions = await asyncio.to_thread(_predict_batch_numpy, batch)
    scores = predictions[0]
    predicted_index = int(np.argmax(scores))

    return _no_cache_response(
        {
            "predicted_class": INDEX_TO_LABEL[predicted_index],
            "confidence": float(scores[predicted_index]),
            "probabilities": _format_probabilities(scores),
        }
    )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        _raise_http_error(400, "No images uploaded.")

    images: List[np.ndarray] = []
    for file in files:
        file_bytes = await file.read()
        _validate_image_bytes(file_bytes, file.filename, file.content_type)
        images.append(_preprocess_image_bytes(file_bytes))

    batch = np.stack(images, axis=0)
    predictions = await asyncio.to_thread(_predict_batch_numpy, batch)

    average_probabilities = np.mean(predictions, axis=0)
    predicted_index = int(np.argmax(average_probabilities))

    return _no_cache_response(
        {
            "num_images": len(files),
            "final_prediction": INDEX_TO_LABEL[predicted_index],
            "confidence": float(average_probabilities[predicted_index]),
            "probabilities": _format_probabilities(average_probabilities),
        }
    )
