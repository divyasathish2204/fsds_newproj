import os
import sys
import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.utils.utills import save_object
from dataclasses import dataclass
from src.DimondPricePrediction.utils.utills import evaluate_model
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
@dataclass
class Modeltrainerconfig():
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer():
    def _init_(self):
        self.model_trainer_config = Modeltrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("spliting dependent and independent variables from train and test datasets")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linearregression':LinearRegression(),
                'Lasso' : Lasso(),
                'Ridge' : Ridge(),
                'Elastic_net' : ElasticNet()
            }
            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print("\n=====================================================\n")
            logging.info(f"Model report:{model_report}")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            print(f"Best_model_name : {best_model_name} and R2 Score :{best_model_score}")
            print('\n=========================================================\n')
            logging.info("best model found :{best_model_name} and r2 score is {best_model_score}")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        except Exception as e:
            logging.info("Exception occured while model training")
            raise CustomException(e,sys)






































class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self):
        pass

    