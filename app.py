import sys
import os
import base64
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from ThermalSolarAnamolyDetection.pipeline.training_pipeline import TrainPipeline
from ThermalSolarAnamolyDetection.constant.application import APP_HOST, APP_PORT

app = FastAPI(
    title="Thermal Solar Anomaly Detection API",
    description="API for detecting anomalies in thermal solar panels using YOLO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup - points to your existing templates/ folder
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "artifacts/model_trainer/best.pt"


class PredictRequest(BaseModel):
    image: str  # base64 encoded image


class PredictResponse(BaseModel):
    image: str  # base64 encoded result image
    detections: int  # number of detections


# Load model at startup if it exists
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Renders your existing templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
def train():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()

        # Reload model after training
        global model
        model = YOLO(MODEL_PATH)

        return {"message": "Training successful!", "model_path": MODEL_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not found. Please train first by calling /train"
            )

        # Decode and save input image
        input_filename = "inputImage.jpg"
        with open(input_filename, "wb") as f:
            f.write(base64.b64decode(request.image))

        # Run inference
        results = model.predict(source=input_filename, conf=0.5, save=True)
        num_detections = len(results[0].boxes)

        # Encode result image to base64
        result_img_path = os.path.join(results[0].save_dir, input_filename)
        with open(result_img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')

        return PredictResponse(image=encoded, detections=num_detections)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)