import os
import sys
import pickle
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models  # Assuming you have this function implemented

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    tfidf_vectorizer_file_path = os.path.join("artifacts", "tfidf_vectorizer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.vectorizer = TfidfVectorizer(max_features=3000)

    def initiate_model_trainer(self, train_df: str, test_df: str) -> float:
        try:
            # Load and preprocess data
            logging.info("Loading and preprocessing data")
            train_df = pd.read_csv(r'artifacts/train_transformed.csv', encoding='ISO-8859-1')
            test_df = pd.read_csv(r'artifacts/test_transformed.csv', encoding='ISO-8859-1')

            # Debug: Print column names
            print("Train DataFrame columns:", train_df.columns)
            print("Test DataFrame columns:", test_df.columns)

            # Fill missing values
            train_df['transformed_text'] = train_df['transformed_text'].fillna('')
            test_df['transformed_text'] = test_df['transformed_text'].fillna('')

            # Extract features and labels
            X_train = self.vectorizer.fit_transform(train_df['transformed_text']).toarray()
            X_test = self.vectorizer.transform(test_df['transformed_text']).toarray()
            y_train = train_df['encoded_target'].values
            y_test = test_df['encoded_target'].values

            # Save the vectorizer
            save_object(
                file_path=self.model_trainer_config.tfidf_vectorizer_file_path,
                obj=self.vectorizer
            )

            logging.info("Starting model training")

            models = {
                "GaussianNB": GaussianNB(),
                "MultinomialNB": MultinomialNB(),
                "BernoulliNB": BernoulliNB()
            }

            params = {
                "GaussianNB": {},
                "MultinomialNB": {},
                "BernoulliNB": {}
            }

            # Evaluate models
            model_report: Dict[str, float] = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model: {best_model_name}")

            # Train and save the best model
            best_model.fit(X_train, y_train)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate the best model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{confusion}")
            logging.info(f"Precision Score: {precision}")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
