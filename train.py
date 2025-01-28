from pathlib import Path 
import os 
import sys
from src.model import iris_classifier
from src.data_processing import load_iris_data, convert_labletoIds
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config
import pandas as pd
import joblib

# Define the root directory of the package
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent

# Add the package root directory to the system path
sys.path.append(str(PACKAGE_ROOT))

def main():
    # Load the data
    data_path = config.filepath
    df = load_iris_data(datapath=data_path)

    print(df.head())
    print(df.variety.value_counts())
    
    # Separate features and target variable
    X = df.drop("variety", axis=1)
    y = df["variety"]
    y = convert_labletoIds(y)

    print("Y after conversion", y.value_counts())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # Initialize the model
    model = iris_classifier()
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_metrics = confusion_matrix(y_test, y_pred)
    
    # Print the results
    print(f"Model Accuracy score is : {accuracy}")
    print(f"Model Confusion score is : \n {conf_metrics}")
    print(f"Model Classification Report : \n {classification_report(y_test, y_pred)}")


    # Save the trained model
    joblib.dump(model, os.path.join(config.SAVE_MODEL_DIR, config.MODEL_NAME))

    print(f"Model saved successfully! at path: {os.path.join(config.SAVE_MODEL_DIR, config.MODEL_NAME)}")
if __name__ == "__main__":
    main()
