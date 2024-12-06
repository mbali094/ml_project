import os
import sys
import dill
import numpy as np
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(file_path:Path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

            file_obj.close()
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):

    try:
        with open(file_path, "rb") as obj:
           return dill.load(obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):

    report= {}
    
    try:

        logging.info("Entered model training.")

        for i in range(len(list(models.values()))):

            model = list(models.values())[i]
            param = param[list(models.keys())[i]]

            gs = GridSearchCV(model, param_grid = param, cv = 3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred= model.predict(X_train)

            train_model_score = r2_score(y_train, y_train_pred)
            print(f"Train score {train_model_score }")
            report[list(models.keys())[i]] = train_model_score
            
            return report

    except Exception as e:
        raise CustomException(e, sys)
