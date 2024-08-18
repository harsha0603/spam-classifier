import os
import sys
import nltk
import string
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "data_preprocessor_vector.pkl")

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords, stemmer):
        self.stopwords = stopwords
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self._transform_text)

    def _transform_text(self, text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        # Remove non-alphanumeric words and stopwords
        y = [word for word in text if word.isalnum() and word not in self.stopwords and word not in string.punctuation]
        y = [self.stemmer.stem(word) for word in y]
        
        return " ".join(y)

from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformation:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.data_transformation_config = DataTransformationConfig()  # Initialize the config

    def get_data_transformer_object(self):
        # Return the preprocessing pipeline or object
        return self.vectorizer

    def initiate_data_transformation(self, train_path, test_path):
        try:
            import os

            train_file_path = os.path.join("artifacts", "train.csv")
            test_file_path = os.path.join("artifacts", "test.csv")

            train_df = pd.read_csv(train_file_path, encoding='ISO-8859-1')
            test_df = pd.read_csv(test_file_path, encoding='ISO-8859-1')

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Starting data transformation")

            # Drop unnecessary columns
            train_df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
            test_df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

            # Rename columns
            train_df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
            test_df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

            # Remove duplicates
            train_df.drop_duplicates(keep='first', inplace=True)
            test_df.drop_duplicates(keep='first', inplace=True)

            # Apply label encoding to the target column
            encoder = LabelEncoder()
            train_df['encoded_target'] = encoder.fit_transform(train_df['target'])
            test_df['encoded_target'] = encoder.transform(test_df['target'])

            # Apply the pipeline on the text column
            X_train = preprocessing_obj.fit_transform(train_df['text']).toarray()
            X_test = preprocessing_obj.transform(test_df['text']).toarray()

            y_train = train_df['encoded_target'].values
            y_test = test_df['encoded_target'].values

            logging.info("Data transformation completed")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed data
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
