# Face Skin Disease Classification API

A FastAPI-based REST API for classifying skin diseases from facial images using a TensorFlow/Keras deep learning model. The model is hosted on Hugging Face Hub and loaded at runtime.

## Features

- **Single Image Prediction** — Classify a single image with confidence scores for all classes
- **Batch Prediction** — Upload multiple images for aggregated prediction
- **Health Check Endpoint** — Simple liveness/readiness probe
- **CORS Support** — Configured for frontend integration
- **Model Auto-Loading** — Downloads model from Hugging Face Hub on startup
- **Docker Ready** — Multi-stage Dockerfile for containerized deployment

## Supported Skin Conditions

The model classifies 12 skin conditions:

| Index | Condition |
|-------|-----------|
| 0 | Eczema |
| 1 | Viral Infections |
| 2 | Melanoma |
| 3 | Atopic Dermatitis |
| 4 | Basal Cell Carcinoma |
| 5 | Melanocytic Nevi |
| 6 | Keratosis-like Lesions |
| 7 | Psoriasis & Lichen Planus |
| 8 | Seborrheic Keratoses |
| 9 | Fungal Infections |

## API Endpoints

### Health Check
```http
GET /health
```
Returns `{"status": "ok"}` if the service is running.

### Single Image Prediction
```http
POST /predict/single
Content-Type: multipart/form-data
```
**Request:** Upload a single image file (max 5MB, JPEG/PNG)

**Response:**
```json
{
  "predicted_class": "Eczema",
  "confidence": 0.9234,
  "probabilities": {
    "Eczema": 0.9234,
    "Viral Infections": 0.0123,
    "Melanoma": 0.0045,
    ...
  }
}
```

### Batch Image Prediction
```http
POST /predict/batch
Content-Type: multipart/form-data
```
**Request:** Upload multiple image files (max 5MB each)

**Response:**
```json
{
  "num_images": 3,
  "final_prediction": "Eczema",
  "confidence": 0.8912,
  "probabilities": {
    "Eczema": 0.8912,
    "Viral Infections": 0.0234,
    ...
  }
}
```

## Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional, for containerized deployment)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_REPO` | Hugging Face repository ID (e.g., `username/model-name`) | **Required** |
| `MODEL_FILENAME` | Model filename in the HF repo | `my_model.h5` |
| `HF_TOKEN` | Hugging Face token (for private repos) | None |
| `PORT` | Server port | `7860` |

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd face-skin-disease-fastapi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variable
export MODEL_REPO="your-username/your-model-repo"

# Run the server
uvicorn app:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

### Docker Deployment

```bash
# Build the image
docker build -t skin-disease-api .

# Run the container
docker run -d \
  -p 7860:7860 \
  -e MODEL_REPO="your-username/your-model-repo" \
  -e HF_TOKEN="your-hf-token" \  # Optional, for private repos
  skin-disease-api
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "7860:7860"
    environment:
      - MODEL_REPO=your-username/your-model-repo
      - HF_TOKEN=${HF_TOKEN}
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Model Requirements

The model must be a TensorFlow/Keras model (`.h5` or SavedModel format) with:
- **Input:** Single RGB image tensor of shape `(batch, height, width, 3)`
- **Output:** Single tensor with 12 logits/probabilities (one per class)
- **Preprocessing:** Uses ResNet preprocessing (`tf.keras.applications.resnet.preprocess_input`)

Upload your trained model to Hugging Face Hub:
```bash
huggingface-cli upload your-username/your-model-repo my_model.h5
```

## Project Structure

```
face-skin-disease-fastapi/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── .dockerignore       # Docker build exclusions
├── .gitignore          # Git exclusions
└── README.md           # This file
```

## CORS Configuration

The API allows requests from:
- `https://face-skin-disease-frontend.devfuze.workers.dev`
- `http://localhost:*` (all local development ports)

To customize, modify the `CORSMiddleware` configuration in `app.py`.

## Performance Notes

- Model loads once at startup (thread-safe singleton)
- Runs on CPU only (`CUDA_VISIBLE_DEVICES=-1`)
- TensorFlow threading limited to 1 thread per op for predictable latency
- Max upload size: 5MB per image
- Input images resized to model's expected dimensions (default 224×224)

## License

MIT License — feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.