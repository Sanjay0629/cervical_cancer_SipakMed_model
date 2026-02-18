"""
FastAPI application for cervical cytology classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

try:
    from api.inference import CervicalInference
except ImportError:
    from inference import CervicalInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cervical Cytology Classification API",
    description="API for classifying cervical cell images into 5 categories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
inference_engine = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    top_3_predictions: List[Dict[str, float]]


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global inference_engine
    
    try:
        # Load configuration
        import yaml
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize inference engine
        model_path = config['api']['model_path']
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            logger.error("Please train the model first")
        else:
            inference_engine = CervicalInference(config)
            logger.info("Model loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    model_loaded = inference_engine is not None
    
    model_info = None
    if model_loaded:
        model_info = {
            "class_names": inference_engine.class_names,
            "num_classes": inference_engine.num_classes,
            "input_shape": inference_engine.target_size
        }
    
    return {
        "status": "online",
        "model_loaded": model_loaded,
        "model_info": model_info
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict cell type from uploaded image
    
    Args:
        file: Uploaded image file (.bmp, .jpg, .jpeg, .png)
        
    Returns:
        PredictionResponse with classification results
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file extension
    allowed_extensions = inference_engine.config['api']['allowed_extensions']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    # Check file size
    max_size = inference_engine.config['api']['max_file_size']
    contents = await file.read()
    
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size / 1024 / 1024:.1f} MB"
        )
    
    try:
        # Load image
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = inference_engine.predict_image(image)
        
        # Format response
        response = {
            "predicted_class": result['predicted_class'],
            "confidence": float(result['confidence']),
            "all_probabilities": {
                class_name: float(prob)
                for class_name, prob in result['all_probabilities'].items()
            },
            "top_3_predictions": [
                {pred['class']: float(pred['probability'])}
                for pred in result['top_3_predictions']
            ]
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict cell types for multiple images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of predictions
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Check file extension
            allowed_extensions = inference_engine.config['api']['allowed_extensions']
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid file type. Allowed: {allowed_extensions}"
                })
                continue
            
            # Load and predict
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            result = inference_engine.predict_image(image)
            
            results.append({
                "filename": file.filename,
                "predicted_class": result['predicted_class'],
                "confidence": float(result['confidence']),
                "all_probabilities": {
                    class_name: float(prob)
                    for class_name, prob in result['all_probabilities'].items()
                }
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results}


@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": inference_engine.class_names,
        "num_classes": inference_engine.num_classes
    }


@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": inference_engine.config['model']['base_model'],
        "input_shape": inference_engine.config['model']['input_shape'],
        "num_classes": inference_engine.num_classes,
        "class_names": inference_engine.class_names,
        "target_size": inference_engine.target_size
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load config for port
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)