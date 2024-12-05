import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_obj(file_path:Path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):

    report= {}
    
    try:

        logging.info("Entered model training.")

        for i in range(len(list(models.values()))):

            model = list(models.values())[i].fit(X_train, y_train)

            y_train_pred= model.predict(X_train)

            train_model_score = r2_score(y_train, y_train_pred)

            report[list(models.keys())[i]] = train_model_score
            
            return report

    except Exception as e:
        raise CustomException(e, sys)
