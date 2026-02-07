import joblib 
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path_r = os.path.join(BASE_DIM,"src","KNN_r.pkl")

model = joblib.load(model_path_r)

class Predict(BaseModel):
    Age : int
    Experience : int 

@app.post("/predict")
def predict(data : Predict):
    input_data = [[data.Age, data.Experience]]
    prediction = model.predict(input_data)

    return{
        "prediction" : int(prediction[0])
    }