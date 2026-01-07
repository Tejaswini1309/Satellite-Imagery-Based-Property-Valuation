# Satellite-Imagery-Based-Property-Valuation

## Overview
This project focuses on predicting house prices using a **multi-modal approach** by incorporating
satellite imagery of house surroundings along with the given tabular features.

Traditional house price prediction models rely only on numerical features such as size, number of
bedrooms, or age of the house. In this project, satellite images are used to capture neighborhood
information such as greenery, road networks, urban density, and proximity to water.

The aim is to understand how surroundings affect the price of a house and, using **Grad-CAM**,
analyze what visual features of the surroundings actually influence the price.

---

## Repository Structure
├── data_fetcher.ipynb
├── eda_and_baseline_models.ipynb
├── model_training.ipynb
├── price_prediction_and_grad_cam.ipynb
├── report.pdf
├── predicted_prices.csv
├── README.md


---

## File Description

### data_fetcher.ipynb
- Used for downloading satellite images using latitude and longitude from the tabular data
- Uses the **ESRI World Imagery API**
- Images are saved with the corresponding **house id** as the filename

### eda_and_baseline_models.ipynb
- Data cleaning and feature engineering
- Exploratory Data Analysis (EDA)
- Experimentation with different baseline models

### model_training.ipynb
- Building of the multi-modal model
- Training the model and evaluating it on the validation set
- Saving the trained model state

### price_prediction_and_grad_cam.ipynb
- Prediction of prices for the test data
- Using Grad-CAM for inference and visual explanation

### 24115050_report.pdf
- Detailed explanation of methodology, experiments, and results

### 24115050_last.csv
- Predicted prices for the test dataset

---

## Dataset
The dataset used for this project was *train1.csv* and *test2.csv* which is provided in the data folder , the data set folder structure required is the google drive is
project_cdc
    ├──train1.csv
    ├──test2.csv
    ├──images 
        ├──train
        ├──test
this is how the folder must look for the data_fetcher.ipynb file , after the images are fetched zip file consisting of train and test images must be made separately , this is to optimise the running time hence the folder must look like this now 
project_cdc
   ├──train1.csv
   ├──test2.csv
   ├──train_images.zip
   ├──test_images.zip
   ├──images
        ├──train
        ├──test
---

## How to Set Up and Run the Project

The project was implemented using **Google Colab**.  
Each `.ipynb` file contains an **Open in Colab** option which can be used to run the notebook on the cloud.

To use Google Colab, a Google account is required.

### Steps:
1. Open the required notebook using the **Open in Colab** option
2. Run the notebook cells sequentially from top to bottom
3. For image extraction, run `data_fetcher.ipynb` first

---

## Model Summary
- **Tabular branch**: Fully connected neural network
- **Image branch**: Pretrained **ResNet-18** 
  - Feature extraction layers are frozen
  - Custom fully connected layers are added
- **Zipcode handling**: Embedding layer for categorical treatment
- Features from all branches are concatenated and passed to a final regression head

---

## Explainability
Grad-CAM is used to analyze which regions of satellite images influence price predictions.
Highlighted regions often include:
- Green areas
- Road networks
- Coastal regions
- Dense residential zones

---

## Evaluation Metrics
- **RMSE**
- **R² Score**

---

## Notes
- Training results may vary slightly due to stochastic behavior (dropout and batch ordering)
- GPU availability affects training time


