"""
app.py -- FastAPI backend with individual endpoints + combined multimodal fusion endpoint.
"""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from grad_cam import (
    EXPLANATIONS,
    UNCERTAIN_THRESHOLD,
    GradCAM,
    build_eval_transform,
    get_device,
    load_model,
    overlay_heatmap,
)
from tabular_model import load_tabular_model, predict_tabular_risk

DEFAULT_CHECKPOINT = "./checkpoints/oa_resnet18_binary.pt"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)

app = FastAPI(title="OA Comprehensive Multimodal Risk Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {
    "model": None,
    "class_names": None,
    "device": None,
    "gradcam": None,
    "tabular_model": None
}

class ClinicalRiskInput(BaseModel):
    age: int = Field(..., ge=18, le=120)
    bmi: float = Field(..., ge=10.0, le=60.0)
    sex: int = Field(..., ge=0, le=1)
    joint_injury_history: int = Field(..., ge=0, le=1)
    occupational_strain: int = Field(..., ge=0, le=2)
    family_history: int = Field(..., ge=0, le=1)
    activity_level: int = Field(..., ge=0, le=2)

@app.on_event("startup")
def load_models_on_startup():
    try:
        _state["tabular_model"] = load_tabular_model()
        print("Tabular clinical model loaded successfully.")
    except Exception as e:
        print(f"Failed to load tabular model: {e}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.exists():
        device = get_device()
        model, class_names = load_model(checkpoint_path, device)
        gradcam = GradCAM(model, target_layer=model.layer4)

        _state["model"] = model
        _state["class_names"] = class_names
        _state["device"] = device
        _state["gradcam"] = gradcam
        print(f"Vision model loaded on {device}. Classes: {class_names}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vision_model_loaded": _state["model"] is not None,
        "tabular_model_loaded": _state["tabular_model"] is not None,
        "device": str(_state["device"]),
        "classes": _state["class_names"],
    }

def encode_image_to_base64(np_image_0_1: np.ndarray) -> str:
    img_uint8 = (np_image_0_1 * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def _run_vision_inference(pil_image: Image.Image):
    device = _state["device"]
    model = _state["model"]
    class_names = _state["class_names"]
    gradcam = _state["gradcam"]

    transform = build_eval_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = logits.argmax(dim=1).item()

    input_tensor.requires_grad_(True)
    cam, probs = gradcam.generate(input_tensor, pred_idx)

    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Identify probability of 'at_risk' specifically
    at_risk_idx = class_names.index("at_risk") if "at_risk" in class_names else 1
    at_risk_prob = float(probs[at_risk_idx])

    overlaid = overlay_heatmap(pil_image, cam)
    heatmap_data_uri = encode_image_to_base64(overlaid)

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "at_risk_prob": round(at_risk_prob, 4),
        "is_uncertain": confidence < UNCERTAIN_THRESHOLD,
        "class_probabilities": {
            class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
        },
        "heatmap_image": heatmap_data_uri,
        "explanation": EXPLANATIONS.get(pred_class, "")
    }

@app.post("/predict-tabular")
def predict_tabular(data: ClinicalRiskInput):
    if _state["tabular_model"] is None:
        raise HTTPException(status_code=503, detail="Tabular risk model not initialized.")
    return JSONResponse(predict_tabular_risk(_state["tabular_model"], data.dict()))

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Vision model not loaded.")
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return JSONResponse(_run_vision_inference(pil_image))

@app.post("/predict-combined")
async def predict_combined(
    clinical_data: str = Form(...),
    image: UploadFile = File(...)
):
    if _state["tabular_model"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="One or both models are not loaded.")

    # 1. Parse clinical JSON data
    try:
        raw_dict = json.loads(clinical_data)
        validated_clinical = ClinicalRiskInput(**raw_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid clinical form payload: {e}")

    # 2. Compute Tabular Risk
    tabular_res = predict_tabular_risk(_state["tabular_model"], validated_clinical.dict())
    p_clinical = tabular_res["risk_probability"]

    # 3. Compute X-Ray Imaging Risk
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    vision_res = _run_vision_inference(pil_image)
    p_xray = vision_res["at_risk_prob"]

    # 4. Multimodal Fusion (40% questionnaire + 60% imaging weight)
    p_combined = (0.40 * p_clinical) + (0.60 * p_xray)

    if p_combined >= 0.60:
        combined_category = "High Overall Risk"
    elif p_combined >= 0.35:
        combined_category = "Moderate Overall Risk"
    else:
        combined_category = "Low Overall Risk"

    return JSONResponse({
        "combined_risk_probability": round(p_combined, 4),
        "combined_risk_category": combined_category,
        "combined_risk_percentage": f"{p_combined:.1%}",
        "clinical_breakdown": tabular_res,
        "imaging_breakdown": vision_res,
        "disclaimer": "This combined score is a clinical screening aid, not a diagnostic verdict."
    })
