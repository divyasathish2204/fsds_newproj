import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open("file_path","wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # training model
            model.fit(x_train,y_train)
            # predicting the test data
            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())][i] = test_model_score
        return report
    except Exception as e:
        logging.info("Exception occured while model training")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open("file_path","rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured while loading obj in utils file")
        raise CustomException(e,sys)
    

    









        