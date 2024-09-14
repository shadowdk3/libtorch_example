import os
import dill
import sys
import torch

from src.exception import CustomException
from src.model import pytorch_resnet50, CNN

class LabelMapping:
    label_to_class = {
        'corn': 0, 
        'orange': 1, 
        'apple': 2, 
        'raddish': 3, 
        'cabbage': 4, 
        'capsicum': 5, 
        'peas': 6, 
        'kiwi': 7,
        'soy_beans': 8, 
        'banana': 9, 
        'garlic': 10, 
        'jalepeno': 11, 
        'bell_pepper': 12, 
        'grapes': 13, 
        'ginger': 14,
        'pomegranate': 15, 
        'cucumber': 16, 
        'sweetcorn': 17, 
        'lettuce': 18, 
        'lemon': 19, 
        'eggplant': 20,
        'chilli_pepper': 21, 
        'potato': 22, 
        'onion': 23, 
        'spinach': 24, 
        'turnip': 25, 
        'mango': 26, 
        'pear': 27,
        'paprika': 28, 
        'sweetpotato': 29, 
        'carrot': 30, 
        'beetroot': 31, 
        'watermelon': 32, 
        'cauliflower': 33,
        'pineapple': 34, 
        'tomato': 35
    }

    class_to_label = {v: k for k, v in label_to_class.items()}
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_checkpoint(file_path):
    try:
        checkpoint = torch.load(file_path)
        return checkpoint
    
    except Exception as e:
        raise CustomException(e, sys)