# Fruits and Vegetables Image Recognition

## Environment 

- Jetson ORIN AGX
- torch
- pytorch

## Install requirments 

```
pip install -U pip
pip install -r requirments.txt
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.15.1-cp38-cp38-manylinux2014_aarch64.whl
```

## Run Script

```
cd /path/to/project
chmod +x srcipt/setup-venv.sh
./srcipt/setup-venv.sh
```


## Download resent50 and convert the model from pytorch to torch

```
mkdir artifacts/
python tools/download_model.py
```

## Dataset

https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

- Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
- Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant

- Train: Contains 100 images per category.
- Test: Contains 10 images per category.
- Validation: Contains 10 images per category.

## Build Project

```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
```

## 

1. load model
2. load custom dataset

## Run program

```
source venv/bin/activate
python src/components/data_ingestion.py
```

