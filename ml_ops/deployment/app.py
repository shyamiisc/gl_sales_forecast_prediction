# Test
import streamlit as st
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
import joblib

# Set the title of the Streamlit app
st.title("Superkart Forecast Revenue")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for property features
product_weight = st.number_input("Product Weight",min_value=0.0,step=1.,format="%.2f")
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0,max_value=1.0,step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0,step=1.,format="%.2f")

product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar","Regular","No Sugar","reg"])
product_type = st.selectbox("Product Type", ["Fruits and Vegetables","Snack Foods","Frozen Foods","Dairy","Household","Baking Goods","Canned","Health and Hygiene","Meat","Soft Drinks","Breads","Hard Drinks","Others","Starchy Foods","Breakfast","Seafood"])

store_size = st.selectbox("Store Size", ["Small","Medium","Large"])
store_city = st.selectbox("Store Size", ["Tier 1","Tier 2","Tier 3"])
store_type = st.selectbox("Store Type", ["Food Mart","Departmental Store","Supermarket Type1","Supermarket Type2"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
            'Product_Weight': product_weight,
        'Product_Allocated_Area': product_allocated_area,
        'Product_MRP': product_mrp,
        'Product_Sugar_Content': product_sugar_content,
        'Product_Type': product_type,
        'Store_Size': store_size,
        'Store_Location_City_Type': store_city,
        'Store_Type': store_type
}])

# Download and load the model
model_path = hf_hub_download(repo_id="shyamgoyal/Sales-Forecast-Prediction", filename="best_sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Make prediction when the "Predict" button is clicked
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
