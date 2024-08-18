import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from src.logger import logging

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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    model_report = {}
    for model_name in models:
        model = models[model_name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_report[model_name] = accuracy
        except Exception as e:
            logging.error(f"Error in evaluating model {model_name}: {str(e)}")
            model_report[model_name] = None
    return model_report


import joblib

def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

