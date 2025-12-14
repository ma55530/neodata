# Neo Orchard QA

Full-stack demo for classifying apples as **good** or **bad** using a FastAPI backend and a React dashboard.

## Back-end (FastAPI)

1. Create/activate a virtual environment in `app/backend`.
2. Install the dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. If you are on Windows without CUDA, install PyTorch + TorchVision with the official wheels, for example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   Pick the URL that fits your OS/GPU from https://pytorch.org/get-started/locally/.
4. Place your trained `model.pt` inside `app/backend/`.
5. Start the API:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## Front-end (React)

1. `cd app/ai-frontend`
2. Install packages: `npm install`
3. Optionally point to a remote API by creating `.env` with `REACT_APP_API_BASE_URL=https://server:8000`
4. Run locally: `npm start`

The dashboard listens on http://localhost:3000 and talks to http://localhost:8000 by default.

## Flow

1. Upload an apple image (JPG/PNG/BMP)
2. Front-end sends it to `/predict`
3. Backend normalizes the image, runs the model, and returns `label`, `confidence`, and a verdict message.
4. UI displays the verdict and keeps a short history of recent checks.
