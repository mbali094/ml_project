import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):

        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data.")

            # DataIngestion.initiate_data_ingestion()
            train_preprocessor = ColumnTransformer(
                [
                    ("ohe", OneHotEncoder(), train_df.select_dtypes("object").columns),
                    ("stdscaler", StandardScaler(), train_df.select_dtypes("number").columns)
                ]
            ) 

            test_preprocessor = ColumnTransformer(
                [
                    ("ohe", OneHotEncoder(), test_df.select_dtypes("object").columns),
                    ("stdscaler", StandardScaler(), test_df.select_dtypes("number").columns)
                ]
            ) 
            train_arr = train_preprocessor.fit_transform(train_df)
            test_arr = test_preprocessor.fit_transform(test_df)

            logging.info("Columns transformation complete.")

            save_obj(self.data_transformation_config.preprocessor_obj_file, train_preprocessor)

            return (train_arr, test_arr, train_preprocessor)
        
        except Exception as e:
            raise CustomException(e, sys)

