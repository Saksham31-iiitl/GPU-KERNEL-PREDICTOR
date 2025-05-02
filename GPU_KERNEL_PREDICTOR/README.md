# GPU Kernel Performance Predictor

The **GPU Kernel Performance Predictor** project is designed to predict the performance of GPU kernels based on various features of the kernels. The model is built using machine learning techniques, specifically a **Random Forest Regressor**, and leverages several features extracted from dataset files.

## Features

- **User-friendly interface**: Allows users to upload their dataset (in CSV format) to get predictions for GPU kernel performance.
- **Data preprocessing**: The tool automatically preprocesses data by scaling features and handling missing values.
- **Prediction Model**: The Random Forest model predicts performance based on input features, leveraging the power of decision trees to provide accurate results.

## Installation

Follow the steps below to set up the project:

### 1. Clone the repository

```bash
git clone https://github.com/your-repo-url/GPU_KERNEL_PREDICTOR.git
cd GPU_KERNEL_PREDICTOR
```


### 2. Set up the environment
It is recommended to use a virtual environment to manage dependencies. You can create one using venv or conda.

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
```

### 3. Install required dependencies
Install the necessary Python packages by running the following command:
```bash
pip install -r requirements.txt
```

### 4. Dataset
Ensure you have the required datasets in the src/data folder:

X.csv: Feature matrix.

y.csv: Target variable for performance prediction.

kernels_clean.csv, model_evaluation.csv, and kernels_dataset.csv: Additional dataset files for the model.

### 5. Model Training
The train_model.py file contains the logic for training the Random Forest model on the X.csv and y.csv datasets. After training, it saves the trained model and scaler to the data/processed directory as:

rf_model.pkl: Random Forest model.

scaler.pkl: Scaler used for data normalization.

feature_columns.pkl: List of feature columns used during training.

To train the model:
```bash 
python train_model.py
```

### 6. Running the Prediction App
The streamlit_app.py file is a Streamlit application for making predictions using the trained model. To run the app:
```bash
streamlit run streamlit_app.py
```

This will launch a web interface where you can upload a CSV file containing kernel data and get performance predictions.


### Requirements
The following Python libraries are required:

pandas

numpy

sklearn

joblib

streamlit

### How It Works
Data Preprocessing: The model reads the feature matrix (X.csv) and the target variable (y.csv) to preprocess the data. This includes scaling the features using StandardScaler.

Model Training: The RandomForestRegressor is trained on the processed data and the model is saved to disk using joblib.

Prediction: After uploading the dataset through the Streamlit app, the model will preprocess the data (scaling), and the trained model will make predictions based on the features in the dataset.
