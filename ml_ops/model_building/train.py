
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

import pandas as pd
import sklearn
# for creating a folder
import os

from huggingface_hub import HfApi

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# To create the pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn import metrics

# To tune different models and standardize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

# for model training, tuning, and evaluation
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib

import mlflow


# function to compute adjusted R-squared
def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# function to compute different metrics to check performance of a regression model
def model_performance_regression(model, predictors, target):
    """
    Function to compute different metrics to check regression model performance

    model: regressor
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    r2 = r2_score(target, pred)  # to compute R-squared
    adjr2 = adj_r2_score(predictors, target, pred)  # to compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, pred))  # to compute RMSE
    mae = mean_absolute_error(target, pred)  # to compute MAE
    mape = mean_absolute_percentage_error(target, pred)  # to compute MAPE

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adj. R-squared": adjr2,
            "MAPE": mape,
        },
        index=[0],
    )

    return df_perf

#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri(public_url)
#mlflow.set_experiment("mlops-training-experiment")

# List of numerical features in the dataset (excluding 'id' as it is an identifier)
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
]

# List of categorical features in the dataset
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
]

# Create a preprocessing pipeline for numerical and categorical features

preprocessor = make_column_transformer(
    (Pipeline([('num_imputer', SimpleImputer(strategy='median')),
               ('scaler', StandardScaler())]), numeric_features),
    (Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent')),
               ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
)

# Perform train-test split
Xtrain_path = "hf://datasets/shyamgoyal/Sales-Forecast-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/shyamgoyal/Sales-Forecast-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/shyamgoyal/Sales-Forecast-Prediction/ytrain.csv"
ytest_path = "hf://datasets/shyamgoyal/Sales-Forecast-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print("Dataset loaded successfully.")

# Define base XGBoost model
xgb_model = XGBRegressor(random_state=42)


# Create pipeline with preprocessing and XGBoost model
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define hyperparameter grid
#Grid of parameters to choose from
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150, 200],    # number of trees to build
    'xgbregressor___max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbregressor___colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbregressor___colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbregressor___learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbregressor___reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.r2_score)

# Run the grid search
grid_obj = GridSearchCV(model_pipeline, param_grid, scoring=scorer,cv=5,n_jobs=-1)
grid_obj = grid_obj.fit(Xtrain, ytrain)

# Set the clf to the best combination of parameters
best_model = grid_obj.best_estimator_

# Fit the best algorithm to the data.
best_model.fit(Xtrain, ytrain)

# Save the model locally
model_path = "best_sales_prediction_model_v1.joblib"
joblib.dump(best_model, model_path)

repo_id = "shyamgoyal/Sales-Forecast-Prediction"
repo_type = "model"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
        path_or_fileobj="best_sales_prediction_model_v1.joblib",
        path_in_repo="./best_sales_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
