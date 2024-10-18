import io
import os
import json
import pickle

import numpy as np # type: ignore
from PIL import Image # type: ignore

import tensorflow as tf # type: ignore
from keras.models import load_model # type: ignore

from fastapi import FastAPI # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore

import numpy as np # type: ignore
from PIL import Image # type: ignore

import tensorflow as tf # type: ignore
from keras.models import load_model # type: ignore

from fastapi import FastAPI, UploadFile, File, HTTPException, Request # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore



# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



app = FastAPI()

# Add CORS Middleware
# origins = [
#     "http://localhost:3000",  # Example: allow access from React app running on localhost:3000
#     "http://127.0.0.1:3000",
#     # Add other origins as needed
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the Crop Disease Prediction model
working_dir = os.path.dirname(os.path.abspath(__file__))
crop_disease_model_path = os.path.join(working_dir, 'Crop_Disease_Prediction_model.h5')
crop_disease_model = load_model(crop_disease_model_path)

# Load the class names for the crop disease prediction model
class_indices_path = os.path.join(working_dir,'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Load the Crop Recommendation System model
crop_recommendation_model_path = os.path.join(working_dir,'DecisionTree.pkl')
with open(crop_recommendation_model_path, 'rb') as model_file:
    crop_recommendation_model = pickle.load(model_file)


def preprocess_image(image):
    # Example preprocessing steps (modify as needed)
    image = image.resize((224, 224))  # Resize to the expected input size of the model
    image_array = np.array(image)     # Convert to numpy array
    image_array = image_array / 255.0 # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a route for the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico") # type: ignore  




# Crop Disease Prediction Route
@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        # Read and process the image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)  # Ensure this function is correctly defined
        
        # Predict the crop disease
        predictions = crop_disease_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        # Return the prediction result as JSON
        return JSONResponse(content={"prediction": predicted_class_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")


# Route for Crop Recommendation
@app.post("/recommend")
async def recommend_crop(data: dict):
    try:
        # Check if required fields are present in data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        # Extract features from the data dictionary
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        # Predict the recommended crop using the loaded model
        recommended_crop = crop_recommendation_model.predict([features])[0]
        
        return JSONResponse(content={'recommended_crop': recommended_crop})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






