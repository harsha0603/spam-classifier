import os
import sys
from dataclasses import dataclass
from typing import Dict
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test) -> float:
        try:
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

            if not model_report:
                raise CustomException("Model report is empty")

            # Get the best model score and name
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
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
