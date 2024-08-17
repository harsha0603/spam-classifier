import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import TextPreprocessor, LabelEncoderPipelineFriendly
from sklearn.feature_extraction.text import TfidfVectorizer  # Import if you use TF-IDF vectorizer

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.vectorizer_path = os.path.join("artifacts", "tfidf_vectorizer.pkl")
        self.preprocessor_path = os.path.join("artifacts", "full_preprocessor_final.pkl")

    def predict(self, features):
        try:
            # Load the model, vectorizer, and preprocessor
            model = load_object(file_path=self.model_path)
            vectorizer = load_object(file_path=self.vectorizer_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Check the types to confirm
            print(f"Model type: {type(model)}")
            print(f"Vectorizer type: {type(vectorizer)}")
            print(f"Preprocessor type: {type(preprocessor)}")

            # Ensure preprocessor is of type TextPreprocessor
            if isinstance(preprocessor, TextPreprocessor):
                preprocessor_transformer = preprocessor
                features_preprocessed = preprocessor_transformer.transform(pd.Series(features))
            else:
                raise CustomException("Preprocessor is not of type TextPreprocessor")

            # Vectorize the preprocessed features
            features_vectorized = vectorizer.transform(features_preprocessed)

            # Make predictions using the model
            preds = model.predict(features_vectorized)

            return preds
        
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)

class CustomData:
    def __init__(self, transformed_text: str):
        self.transformed_text = transformed_text

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "transformed_text": [self.transformed_text],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
