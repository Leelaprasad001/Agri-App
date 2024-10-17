import io
import os
import json
import pickle
import numpy as np
import pandas as pd  # Import pandas for DataFrame
from PIL import Image


import warnings
warnings.filterwarnings("ignore") # Suppress warnings

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.models import load_model    # type: ignore

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the Crop Disease Prediction model
working_dir = os.path.dirname(os.path.abspath(__file__))
crop_disease_model_path = os.path.join(working_dir, 'Models', 'Crop Disease Prediction model.h5')
crop_disease_model = load_model(crop_disease_model_path)

# Load the class names for the crop disease prediction model
class_indices_path = os.path.join(working_dir, 'Data', 'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Load the Crop Recommendation System model
crop_recommendation_model_path = os.path.join(working_dir, 'Models', 'DecisionTree.pkl')
with open(crop_recommendation_model_path, 'rb') as model_file:
    crop_recommendation_model = pickle.load(model_file)

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the expected input size of the model
    image_array = np.array(image)     # Convert to numpy array
    image_array = image_array / 255.0 # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


class CropRecommendationData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Serve the favicon.ico file
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(working_dir, 'static', 'favicon.ico'))


# Crop Disease Prediction Route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        
        predictions = crop_disease_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        return JSONResponse(content={"prediction": predicted_class_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")


# Route for Crop Recommendation
@app.post("/recommend")
async def recommend_crop(data: CropRecommendationData):
    try:
        # Create a DataFrame with appropriate feature names
        features = pd.DataFrame({
            'N': [data.N],
            'P': [data.P],
            'K': [data.K],
            'temperature': [data.temperature],
            'humidity': [data.humidity],
            'ph': [data.ph],
            'rainfall': [data.rainfall]
        })

        # Predict the recommended crop using the loaded model
        recommended_crop = crop_recommendation_model.predict(features)[0]

        return JSONResponse(content={'recommended_crop': recommended_crop})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)