# LinfomaClassification

This repository contains scripts to train and evaluate Convolutional Neural Networks (CNNs) using pre-trained models such as MobileNetV2, ResNet50, VGG16, and InceptionV3. The scripts are designed to iterate over various input image sizes, evaluate model performance, and measure inference time.

## Setup

    git clone https://github.com/julenbhy/LinfomaClassification

  Copy your datased to `dataeset\`. Structure should follow:

     dataset
      ├── ...
      ├── class1  
      │   ├── img1.png
      │   ├── img2.png
      │   ├── ...
      ├── class2  
      │   ├── img1.png
      │   ├── img2.png
      │   ├── ...
      ├── ...


## Configuration

In the `models_config.py` file, you can select the models to be executed and their respective resolutions.

Other parameters such as paths or training parameters can be configured in the first cell of `MultiModelTrainer.ipynb`.

## Run
Run `MultiModelTrainer.ipynb`

## Results
The final results with the values obtained for all models and resolutions will be stored in [`logs/test_results.py`](https://github.com/julenbhy/LinfomaClassification/blob/main/logs/test_results.csv). 

Multiple comparative graphs of these values will also be generated in the `plots/` directory.

Results:

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_loss.png"> 

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_accuracy.png"> 

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_precision.png"> 

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_recall.png"> 

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_loss.png"> 

<img width="1000" alt="portfolio_view" src="https://github.com/julenbhy/LinfomaClassification/blob/main/plots/comparison_test_time.png"> 

