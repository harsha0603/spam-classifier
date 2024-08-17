import os
import sys
import nltk
import string
import pickle
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "full_preprocessor.pkl")

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
        
        y = [word for word in text if word.isalnum()]
        y = [word for word in y if word not in self.stopwords and word not in string.punctuation]
        y = [self.stemmer.stem(word) for word in y]
        
        return " ".join(y)

class LabelEncoderPipelineFriendly(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates a complete pipeline with text preprocessing and label encoding.
        '''
        try:
            pipeline = Pipeline([
                ('text_preprocessor', TextPreprocessor(stopwords.words('english'), PorterStemmer())),
                ('label_encoder', LabelEncoderPipelineFriendly())
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

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

            # Apply the pipeline on text and target columns
            train_df['transformed_text'] = preprocessing_obj.named_steps['text_preprocessor'].transform(train_df['text'])
            train_df['encoded_target'] = preprocessing_obj.named_steps['label_encoder'].fit_transform(train_df['target'])

            test_df['transformed_text'] = preprocessing_obj.named_steps['text_preprocessor'].transform(test_df['text'])
            test_df['encoded_target'] = preprocessing_obj.named_steps['label_encoder'].transform(test_df['target'])

            logging.info("Data transformation completed")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_df,
                test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
