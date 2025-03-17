import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

from src.features.feature_definitions import feature_build

app = FastAPI()

class PredictionInput(BaseModel):
    vendor_id: float
    pickup_datetime: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float

# Load the pre-trained RandomForest model
model_path = "model/model.joblib"  # Ensure model is in /app/model/model.joblib
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    features = {
        'vendor_id': input_data.vendor_id,
        'pickup_datetime': input_data.pickup_datetime,
        'passenger_count': input_data.passenger_count,
        'pickup_longitude': input_data.pickup_longitude,
        'pickup_latitude': input_data.pickup_latitude,
        'dropoff_longitude': input_data.dropoff_longitude,
        'dropoff_latitude': input_data.dropoff_latitude,
        'store_and_fwd_flag': input_data.store_and_fwd_flag
    }
    
    features = pd.DataFrame(features, index=[0])
    features = feature_build(features, 'prod')
    prediction = model.predict(features)[0].item()
    
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Change host to "0.0.0.0" for external access
