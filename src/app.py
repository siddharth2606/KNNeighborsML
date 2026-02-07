import joblib 
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIM,"src","KNN.pkl")

model = joblib.load(model_path)

class Predict(BaseModel):
    Age : int
    EstimatedSalary : int 

@app.post("/predict")
def predict(data : Predict):
    input_data = [[data.Age, data.EstimatedSalary]]
    prediction = model.predict(input_data)

    return{
        "prediction" : int(prediction[0])
    }