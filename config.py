import pathlib
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scale_ml_project as sf

# Define the root directory of the project
ROOT_DIR = pathlib.Path(sf.__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

#verify the root directory
print("ROOT_DIR: ", ROOT_DIR)

# Define the directory where raw data is stored
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')

# Define the name of the data file
file_name = "iris.csv"

# Define the full path to the data file
filepath = os.path.join(DATA_DIR, file_name)

# Define the name of the model file
MODEL_NAME = "iris_model.joblib"

# Define the directory where the model will be saved
SAVE_MODEL_DIR = os.path.join(ROOT_DIR, "models")
