import os
from src.utils import load_object

# Define the paths to your artifacts
model_path = os.path.join("artifacts", "model.pkl")
vectorizer_path = os.path.join("artifacts", "tfidf_vectorizer.pkl")
preprocessor_path = os.path.join("artifacts", "full_preprocessor_final.pkl")

# Load the objects
model = load_object(file_path=model_path)
vectorizer = load_object(file_path=vectorizer_path)
preprocessor = load_object(file_path=preprocessor_path)

# Print the types of the objects
print(f"Model type: {type(model)}")
print(f"Vectorizer type: {type(vectorizer)}")
print(f"Preprocessor type: {type(preprocessor)}")
