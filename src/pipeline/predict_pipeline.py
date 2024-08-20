import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "vectorizer.pkl")

    def predict(self, features):
        try:
            # Load the model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Ensure preprocessor is of type TfidfVectorizer or custom preprocessor
            if not hasattr(preprocessor, 'transform'):
                raise CustomException("Preprocessor object does not have a transform method")

            # Apply the preprocessing and vectorization
            features_vectorized = preprocessor.transform(features)

            # Make predictions using the model
            preds = model.predict(features_vectorized)

            return preds
        
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)
