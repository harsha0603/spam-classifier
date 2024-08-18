import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the raw data
            df = pd.read_csv('notebook/data/spam.csv', encoding='ISO-8859-1')
            logging.info('Read the dataset as dataframe')

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Data Ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Save transformed data to files
    train_data_transformed_path = 'artifacts/train_transformed_final.csv'
    test_data_transformed_path = 'artifacts/test_transformed_final.csv'
    
    # Model Training
    model_trainer = ModelTrainer()
    best_model_score = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
