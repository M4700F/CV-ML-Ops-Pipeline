# 🌡️ Thermal Solar Anomaly Detection

An end-to-end MLOps pipeline for detecting anomalies in thermal solar panel images using YOLOv26 and Ultralytics.

🚀 **Live Demo**: [https://cv-ml-ops-pipeline.onrender.com](https://cv-ml-ops-pipeline.onrender.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Pipeline Stages](#pipeline-stages)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Overview

This project detects 8 types of solar panel anomalies from thermal images:

| Class | Description |
|-------|-------------|
| `MultiByPassed` | Multiple bypassed cells |
| `MultiDiode` | Multiple diode faults |
| `MultiHotSpot` | Multiple hotspot defects |
| `SingleByPassed` | Single bypassed cell |
| `SingleDiode` | Single diode fault |
| `SingleHotSpot` | Single hotspot defect |
| `StringOpenCircuit` | Open circuit in string |
| `StringReversedPolarity` | Reversed polarity in string |

---

## Project Structure

```
CV-ML-Ops-Pipeline/
├── app.py                               # FastAPI application entry point
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .project-root                        # Required by from_root package
├── .github/
│   └── workflows/
│       └── deploy.yml                   # GitHub Actions CI/CD pipeline
├── templates/
│   └── index.html                       # Frontend UI
├── artifacts/                           # Auto-generated during pipeline run
│   ├── data_ingestion/
│   │   └── feature_store/               # Downloaded dataset
│   ├── data_validation/
│   │   └── status.txt                   # Validation results
│   └── model_trainer/
│       ├── best.pt                      # Final trained model
│       └── custom_data.yaml             # Generated training config
└── ThermalSolarAnamolyDetection/
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   └── model_trainer.py
    ├── pipeline/
    │   └── training_pipeline.py
    ├── entity/
    │   ├── config_entity.py
    │   └── artifacts_entity.py
    ├── constant/
    │   ├── training_pipeline.py
    │   └── application.py
    ├── logger/
    └── exception/
```

---

## Dataset

- **Source**: [Kaggle — Solar Panel Dataset](https://www.kaggle.com/datasets/pkdarabi/solarpanel)
- **Downloaded automatically** via `kagglehub` on first pipeline run
- **Classes**: 8 anomaly types
- **Splits**: train / valid / test

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/CV-ML-Ops-Pipeline.git
cd CV-ML-Ops-Pipeline
```

### 2. Create and activate conda environment

```bash
conda create -n thermo python=3.11 -y
conda activate thermo
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API credentials

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

> Get your API key from [kaggle.com](https://www.kaggle.com) → Profile → Settings → API → Create New Token

---

## Running the Project

Start the FastAPI server locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then open your browser at:

```
http://localhost:8080
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/train` | Trigger full training pipeline |
| `POST` | `/predict` | Run inference on an image |
| `GET` | `/docs` | Auto-generated Swagger API docs |

### `/predict` request format

```json
{
  "image": "<base64_encoded_image_string>"
}
```

### `/predict` response format

```json
{
  "image": "<base64_encoded_result_image>",
  "detections": 3
}
```

---

## Pipeline Stages

### 1. Data Ingestion
- Downloads the solar panel dataset from Kaggle using `kagglehub`
- Copies data into `artifacts/data_ingestion/feature_store/`

### 2. Data Validation
- Checks that all required files exist: `train/`, `valid/`, `test/`, `data.yaml`
- Writes validation status to `artifacts/data_validation/status.txt`
- Stops the pipeline if validation fails

### 3. Model Training
- Generates a custom `data.yaml` with correct absolute paths
- Auto-downloads `yolo26n.pt` pretrained weights if not present
- Trains using Ultralytics YOLOv26
- Saves the best weights to `artifacts/model_trainer/best.pt`

---

## Deployment

This project is deployed on **Render** with a fully automated CI/CD pipeline via **GitHub Actions**.

### How it works

```
Push to main → GitHub Actions triggers → Docker image built → Deployed to Render
```

### Deploy your own instance

1. Fork this repository
2. Sign up at [render.com](https://render.com) and connect your GitHub account
3. Create a **New Web Service** → select your forked repo → Render auto-detects the Dockerfile
4. Add environment variables in Render dashboard under **Environment**:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

5. In Render → **Settings** → **Deploy Hook** → copy the URL
6. Add it as `RENDER_DEPLOY_HOOK` secret in your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
7. Push any change to `main` — GitHub Actions will auto-deploy

> **Note**: On Render's free tier, the service sleeps after 15 minutes of inactivity and takes ~30 seconds to wake up on the next request.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `ultralytics` | YOLOv26 model training & inference |
| `kagglehub` | Dataset download |
| `FastAPI` | REST API server |
| `uvicorn` | ASGI server |
| `python-dotenv` | Environment variable management |
| `PyYAML` | YAML config handling |
| `PyTorch` | Deep learning backend |
| `Docker` | Containerization |
| `GitHub Actions` | CI/CD pipeline |
| `Render` | Cloud hosting |

---

## Training Configuration

Defined in `ThermalSolarAnamolyDetection/constant/training_pipeline.py`:

| Parameter | Value |
|-----------|-------|
| Model | `yolo26n.pt` |
| Epochs | `50` |
| Batch Size | `16` |
| Image Size | `640` |