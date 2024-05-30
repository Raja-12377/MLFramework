import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Binarizer, StandardScaler, Normalizer
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from collections import Counter


from sklearn.preprocessing import MinMaxScaler, Binarizer, StandardScaler, Normalizer

def perform_feature_scaling(data):
    st.subheader("Feature Scaling (Numerical Columns Only)")

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    X_numerical = data[numerical_cols]
    X_categorical = data[categorical_cols]

    scaling_methods = ['StandardScaler', 'MinMaxScaler', 'Binarizer', 'Normalizer']
    selected_scaling_method = st.selectbox("Select Feature Scaling Method", scaling_methods)
    st.write(f"Performing {selected_scaling_method} Feature Scaling on numerical features...")

    if selected_scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif selected_scaling_method == 'Binarizer':
        scaler = Binarizer()
    elif selected_scaling_method == 'StandardScaler':
        scaler = StandardScaler()
    elif selected_scaling_method == 'Normalizer':
        scaler = Normalizer()

    scaled_data = scaler.fit_transform(X_numerical)
    scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_cols)

    # Combine scaled numerical columns with original categorical columns
    combined_data = pd.concat([scaled_data_df, X_categorical], axis=1)

    st.success(f"{selected_scaling_method} Feature Scaling completed.")
    st.write("Scaled data:")
    st.write(combined_data)

    return combined_data
    


