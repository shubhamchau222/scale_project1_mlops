from sklearn.linear_model import LogisticRegression
import joblib

class iris_classifier:
    def __init__(self):
        # Initialize the Logistic Regression model with a maximum of 1000 iterations
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        """
        Train the Logistic Regression model using the provided features and labels.
        
        Parameters:
        X (array-like): Training data
        y (array-like): Target values
        """
        # Fit the model to the training data
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        X (array-like): Data to predict
        
        Returns:
        array: Predicted class labels
        """
        # Predict the class labels for the input data
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.
        
        Parameters:
        X (array-like): Data to predict
        
        Returns:
        array: Predicted class probabilities
        """
        # Predict the class probabilities for the input data
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        filepath (str): Path to save the model
        """
        # Save the model to the specified filepath
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        filepath (str): Path to load the model from
        """
        # Load the model from the specified filepath
        self.model = joblib.load(filepath)
