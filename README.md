# 🌡️ Thermal Solar Anomaly Detection

An end-to-end MLOps pipeline for detecting anomalies in thermal solar panel images using YOLOv26 and Ultralytics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Pipeline Stages](#pipeline-stages)
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
├── app.py                          # FastAPI application entry point
├── requirements.txt
├── templates/
│   └── index.html                  # Frontend UI
├── artifacts/                      # Auto-generated during pipeline run
│   ├── data_ingestion/
│   │   └── feature_store/          # Downloaded dataset
│   ├── data_validation/
│   │   └── status.txt              # Validation results
│   └── model_trainer/
│       ├── best.pt                 # Final trained model
│       └── custom_data.yaml        # Generated training config
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
    ├── logger.py
    └── exception.py
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
git clone <your-repo-url>
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

Start the FastAPI server:

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
- Generates a custom `data.yaml` with correct local paths
- Auto-downloads `yolo26n.pt` pretrained weights if not present
- Trains using Ultralytics YOLOv26
- Saves the best weights to `artifacts/model_trainer/best.pt`

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

---

## Training Configuration

Defined in `ThermalSolarAnamolyDetection/constant/training_pipeline.py`:

| Parameter | Value |
|-----------|-------|
| Model | `yolo26n.pt` |
| Epochs | `50` |
| Batch Size | `16` |
| Image Size | `640` |


constant
entity
components
pipelines
app.py