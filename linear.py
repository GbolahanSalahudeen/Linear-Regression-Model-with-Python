import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

file_path = "sales_data.csv"

df = pd.read_csv(file_path)



feature_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Promotion', 'Competitor Pricing',\
                'Epidemic']
X = df[feature_cols]
y = df["Demand"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Title
st.title("Demand Prediction App")

# Input fields
st.subheader("Enter the values for each feature:")
user_input = []

for i, feature in enumerate(features):
    value = st.number_input(label=f"{feature}", step=1.0, key=f"{feature}_{i}")
    user_input.append(value)
    
# Prediction
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    st.success(f"Predicted Demand: {prediction:.2f}")