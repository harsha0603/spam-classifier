import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
from typing import Dict
from sklearn.metrics import accuracy_score

def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, any], param: Dict[str, Dict[str, any]]) -> Dict[str, float]:
    """
    Evaluates different models and returns their performance metrics.
    
    Args:
    X_train: Training feature data.
    y_train: Training target data.
    X_test: Testing feature data.
    y_test: Testing target data.
    models: Dictionary of models to evaluate.
    param: Dictionary of hyperparameters for each model.

    Returns:
    Dictionary with model names and their respective accuracy scores.
    """
    model_report = {}

    for model_name, model in models.items():
        try:
            # If hyperparameters are specified, set them
            if param[model_name]:
                model.set_params(**param[model_name])
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            model_report[model_name] = accuracy

        except Exception as e:
            print(f"Error occurred while evaluating {model_name}: {e}")

    return model_report

import joblib

def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

