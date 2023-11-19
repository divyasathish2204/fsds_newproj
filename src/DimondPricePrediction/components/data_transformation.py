import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.DimondPricePrediction.utils.utills import save_object

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformed(self):
        
        try:
            logging.info("data transformation initiated")
            # define which column should be ordinal encoded and which should be scaled
            categorical_cols = ["cut","color","clarity"]
            numerical_cols = ["carat","depth","table","x","y","z"]
            # define custom ranking for each ordinary variables
            cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline initiated")
            ## numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            # categorical_pipeline
            cat_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy="most frequent")),
                    ('Encoder',OrdinalEncoder([cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
             ]
             )

            return preprocessor
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)
        

        def initialize_datatransformation(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                logging.info("i have read the train data")
                logging.info("i have read the test data")
                logging.info(f'train dataframe head : \n {train_df.head().to_string()}')
                logging.info(f'test dataframe head :\n {test_df.head().to_string()}')
                preprocessor_obj = self.get_data_transformed
                target_column_name = 'price'
                drop_columns = [target_column_name,'id']
                input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
                target_feature_train_df = train_df[target_column_name]
                input_feature_test_df = test.df.drop(columns=drop_columns,axis=1)
                target_feature_test_df = test_df[target_column_name]

                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
                logging.info("Applying preprocessor obj on training and testing datasets")
                train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor_obj
                )

                logging.info("preprocessing pickle file saved")
                return(
                    train_arr,
                    test_arr
                )
            except Exception as e:
                logging.info("An exception ocuured while data transformation")
                print(e,sys)



            
            

                




















    





















