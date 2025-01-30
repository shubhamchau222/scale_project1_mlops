from fastapi import FastAPI  # Import FastAPI for creating the web application
from pydantic import BaseModel  # Import BaseModel for data validation
import uvicorn  # Import uvicorn for running the ASGI server
import joblib  # Import joblib for loading the machine learning model
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import os  # Import os for interacting with the operating system
import sys  # Import sys for system-specific parameters and functions
from pathlib import Path  # Import Path for handling filesystem paths
import requests  # Import requests for making HTTP requests

# Define the directory where the model is stored
model_dir = Path(__file__).resolve().parent / "models"
# Define the source directory and add it to the system path
src = Path(__file__).resolve().parent / "src"
sys.path.append(str(src))

# Define the path to the model file and load the model
model_path = model_dir / "iris_model.joblib"
model = joblib.load(model_path)

# Initialize the FastAPI application
app = FastAPI()

# Definet the root endpoint to return the welcome message
@app.get("/")
def home():
    return {"message": "Welcome to the Iris Classification API!"}

# Define the pydentic model for Input data 

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    # "sepal_length","sepal_width","petal_length","petal_width","variety"


#define the endpoint for making predictions
@app.post("/predict")
def predict_iris(data: IrisData):

    """predict the species based on given number"""
    data = data.dict()
    # print("DTTTTTTTTTTTTTTTTTT : ", data)
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

    #convert data to dataframe
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    print("Input data inserted by the user for predictions is :")
    print(input_data)

    # Make prediction
    prediction= model.predict(input_data)[0]

    # map predinction to the class label
    original_dict = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    reversed_dict = {v: k for k, v in original_dict.items()}
    return {"prediction": str(reversed_dict[prediction])}

# Run the FASTapi application with uvicorn
if __name__ == "__main__":
    uvicorn.run( app, 
                host="127.0.0.1", 
                port=8000
                )


