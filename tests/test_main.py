# tests/test_main.py

from fastapi.testclient import TestClient
import sys
import pathlib

# Adjust the path so the application can be imported
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from main import app  # Import your FastAPI app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    # Check for a successful response
    assert response.status_code == 200
    # Confirm the response content matches expectation
    assert response.json() == "Welcome to Fast API Application"

def test_predict_endpoint():
    # Create a sample payload with valid Iris features
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    # Verify the response status code is 200 (OK)
    assert response.status_code == 200
    # Confirm that the response contains the prediction key
    data = response.json()
    assert "species prediction" in data
    # Optionally, check that the prediction is one of the expected species
    assert data["species prediction"] in ["Setosa", "Versicolor", "Virginica"]

def test_predict_endpoint_invalid_data():
    # Test with missing or invalid fields
    payload = {
        "sepal_length": "invalid",  # providing invalid data type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    # Expecting a validation error (422 Unprocessable Entity)
    assert response.status_code == 422
