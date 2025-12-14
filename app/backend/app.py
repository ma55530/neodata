import io
import sys
import os
import tempfile
from pathlib import Path
from typing import Dict

print(f"[startup] Python executable: {sys.executable}")

import logging
import signal
import traceback

from combined_detector_service import CombinedDetectorService
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from ultralytics import YOLO


try:
    from pillow_heif import register_heif_opener, read_heif, open_heif

    register_heif_opener()
    HEIF_SUPPORT = True
    print("[startup] pillow-heif registered successfully.")
except ImportError:
    HEIF_SUPPORT = False
    print("[startup] pillow-heif is not installed; HEIC conversion endpoint disabled.")
except Exception as e:
    HEIF_SUPPORT = False
    print(f"[startup] Failed to register pillow-heif: {e}")

APP_TITLE = "Apple Quality Classifier"
MODEL_FILENAME = "model.pt"
IMAGE_SIZE = 224
CLASS_LABELS: Dict[int, str] = {
    0: "pass",
    1: "fail",
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_FILENAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def is_heic_upload(upload: UploadFile) -> bool:
    """Best-effort detection whether a file is HEIC/HEIF."""
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    return "hei" in content_type or filename.endswith((".heic", ".heif"))


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor ready for inference."""
    return preprocess_pipeline(image)


def load_model():
    """Load the trained model using Ultralytics YOLO class."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Could not find model checkpoint at {MODEL_PATH}. "
            "Make sure 'model.pt' is exported and copied to the backend folder."
        )
    
    print(f"[startup] Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(str(MODEL_PATH))
        print(f"[startup] Model loaded successfully. Task: {model.task}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

try:
    model = load_model()
except RuntimeError as err:
    model = None
    print(f"[startup] Model failed to load: {err}")

try:
    combined_detector = CombinedDetectorService(BASE_DIR)
    print("[startup] Combined detector service initialized.")
except Exception as exc:
    combined_detector = None
    print(f"[startup] Combined detector unavailable: {exc}")


app = FastAPI(title=APP_TITLE)


# configure basic logging to capture startup/shutdown traces
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Signal handlers to log termination reasons from the platform (SIGTERM/SIGINT)
def _handle_termination(signum, frame):
    logger.info(f"Received termination signal: {signum}. Dumping stack and exiting.")
    try:
        traceback.print_stack(frame)
    except Exception:
        logger.exception("Error printing stack")
    # do not call sys.exit here; let FastAPI/uvicorn handle graceful shutdown


signal.signal(signal.SIGTERM, _handle_termination)
signal.signal(signal.SIGINT, _handle_termination)


@app.on_event("startup")
def _log_startup():
    logger.info("Application startup complete. PID=%s", os.getpid())


@app.on_event("shutdown")
def _log_shutdown():
    logger.info("Application shutdown event triggered. PID=%s", os.getpid())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEGMENTATION_DIR = BASE_DIR / "ai" / "segmented_output_improved"
if SEGMENTATION_DIR.exists():
    app.mount("/segmentation", StaticFiles(directory=SEGMENTATION_DIR), name="segmentation")
else:
    print(f"[startup] segmented_output_improved not found at {SEGMENTATION_DIR}, overlay previews disabled.")


@app.get("/health")
async def health_check():
    """Simple health check endpoint for uptime monitoring."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "combined_detector_ready": combined_detector is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.") from exc

    # Use Ultralytics predict method which handles preprocessing and postprocessing
    results = model.predict(image, device=DEVICE, verbose=False)
    result = results[0]

    # Determine if it's classification or detection
    if result.probs is not None:
        # Classification task
        probs = result.probs.data
        predicted_idx = probs.argmax()
        confidence = probs[predicted_idx]
        predicted_label = result.names[predicted_idx.item()]
    elif result.boxes is not None:
        # Detection task - infer pass/fail based on detections
        # Assuming if we detect "fail" class, it's a fail.
        # We need to know which class ID corresponds to what.
        # Let's use the model's internal names.
        
        detected_classes = [result.names[int(cls)] for cls in result.boxes.cls]
        
        # Simple logic: if "fail" or "defect" is in detected classes, then fail.
        # Adjust this logic based on your actual class names!
        is_fail = any("fail" in cls.lower() or "defect" in cls.lower() for cls in detected_classes)
        
        predicted_label = "fail" if is_fail else "pass"
        
        # Calculate confidence as max confidence of detected boxes, or 1.0 if nothing detected (pass)
        if len(result.boxes.conf) > 0:
            confidence = result.boxes.conf.max()
        else:
            confidence = torch.tensor(1.0)
            
    else:
        # Fallback or unknown task
        predicted_label = "unknown"
        confidence = torch.tensor(0.0)

    verdict_message = (
        "Element zadovoljava KFK standard i može na gradilište."
        if predicted_label == "pass"
        else "Element ne prolazi QC — treba dodatnu inspekciju ili popravak."
    )

    defects = []
    # If detection, list defects
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = result.names[cls_id]
            conf = float(box.conf)
            defects.append({
                "type": cls_name,
                "confidence": round(conf, 4),
                "box": box.xyxy[0].tolist()
            })

    return {
        "label": predicted_label,
        "confidence": round(confidence.item(), 4),
        "verdict": verdict_message,
        "defects": defects,
    }


@app.post("/detect/combined")
async def detect_combined(file: UploadFile = File(...)):
    if combined_detector is None:
        raise HTTPException(status_code=503, detail="Kombinirani detektor trenutno nije dostupan.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Datoteka nema naziv. Ponovno izvezite fotografiju.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    probe = None
    try:
        probe = Image.open(io.BytesIO(image_bytes))
        probe.verify()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Ne mogu pročitati poslanu sliku.") from exc
    except Exception as exc:  # pragma: no cover - sanity fallback
        raise HTTPException(status_code=400, detail="Format datoteke nije podržan.") from exc
    finally:
        if probe:
            probe.close()

    try:
        report = combined_detector.analyze_by_filename(file.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        print(f"[combined] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail="Kombinirani detektor je prijavio grešku.") from exc

    return report


@app.post("/convert/heic")
async def convert_heic(file: UploadFile = File(...)):
    if not HEIF_SUPPORT:
        raise HTTPException(status_code=503, detail="HEIC dekoder nije dostupan na serveru.")

    if not is_heic_upload(file):
        raise HTTPException(status_code=400, detail="Ovaj endpoint pretvara samo HEIC/HEIF datoteke.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    print(f"DEBUG: Received {len(image_bytes)} bytes. Header: {image_bytes[:16].hex()}")

    # Save to temp file to avoid BytesIO issues with C-libraries
    with tempfile.NamedTemporaryFile(delete=False, suffix=".heic") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # Try opening from file path
        image = Image.open(tmp_path)
        image.load() # Force load
    except Exception as exc:
        print(f"HEIC conversion error with Image.open(path): {exc}")
        try:
            # Fallback with pillow_heif directly on file
            print("DEBUG: Attempting fallback with pillow_heif.open_heif on file...")
            heif_file = open_heif(tmp_path)
            
            # Iterate to find the primary image or just take the first one
            for img in heif_file:
                image = Image.frombytes(
                    img.mode,
                    img.size,
                    img.data,
                    "raw",
                )
                break # Just take the first one
            print("DEBUG: Fallback successful.")
        except Exception as e:
            print(f"DEBUG: Fallback failed: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"Ne mogu pročitati HEIC datoteku: {exc} | {e}") from exc
    
    # Clean up temp file
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    buffer = io.BytesIO()

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    buffer.seek(0)

    safe_name = Path(file.filename or "converted").stem or "converted"
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}.jpg"'}

    return StreamingResponse(buffer, media_type="image/jpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
