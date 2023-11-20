from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer
import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
import pandas as pd
obj = DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()
obj1 = DataTransformation()
train_arr,test_arr=obj1.initialize_datatransformation(train_data_path,test_data_path)
model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)
