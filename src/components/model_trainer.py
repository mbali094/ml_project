import os
import sys
from pathlib import Path
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj, evaluate_model
from src.exception import CustomException
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the data.")

            X_train, y_train, X_test, y_test =(
                train_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models ={
            'linearregression':LinearRegression(),
            'Ridge':Ridge(),
            'Lasso':Lasso(),
            'decisiontree': DecisionTreeRegressor(),
            'randomforestregressor':RandomForestRegressor(),
            'adaboosteregressor':AdaBoostRegressor(),        
            'kneighborregressor':KNeighborsRegressor(),
            'catboostregressor':CatBoostRegressor(),
            'xgbregressor':XGBRegressor()
             }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models =models)
            
            logging.info("Model training is over.")

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            
            logging.info("Found a best model on both training and test set.")

            save_obj(file_path = self.model_trainer_config.trained_model_file_path, obj=best_model)

            y_test_pred= best_model.predict(X_test)
            
            r2_test_score = r2_score(y_test, y_test_pred)

            return r2_test_score

        except Exception as e:
            raise CustomException(e, sys)