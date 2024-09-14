import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import src.utils

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data', "train")
    test_data_path: str=os.path.join('data', "test")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or compont")
        try:
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            return CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_loader, test_loader = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    best_epoch, best_accuracy, best_loss = modeltrainer.initiate_model_trainer(train_loader, test_loader)
    
    logging.info(f'The best model is trained at {best_epoch} epoch: accuracy - {best_accuracy}, loss - {best_loss}')
    