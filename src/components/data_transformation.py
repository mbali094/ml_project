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

            input_train_df = train_df.drop(columns=["math score"])
            target_train_series=train_df["math score"]

            
            input_test_df = test_df.drop(columns=["math score"])
            target_test_series=test_df["math score"]

            preprocessor = ColumnTransformer(
                [
                    ("ohe", OneHotEncoder(), input_train_df.select_dtypes("object").columns),
                    ("stdscaler", StandardScaler(), input_train_df.select_dtypes("number").columns)
                ]
            ) 

            train_arr = preprocessor.fit_transform(input_train_df)
            test_arr = preprocessor.transform(input_test_df)

            logging.info("Columns transformation complete.")

            save_obj(self.data_transformation_config.preprocessor_obj_file, preprocessor)

            return (train_arr, test_arr, preprocessor)        

        except Exception as e:
            raise CustomException(e, sys)

