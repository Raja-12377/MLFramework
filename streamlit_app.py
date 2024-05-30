import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from Missing_treatment import handle_missing_values
from outliers_treatment import preprocess_outliers
from f_scaling import perform_feature_scaling
from encoding_treatment import preprocess_encoding
from distribution import distribution
from f_selection import feature_selection
from model import model_training
from unsup_f_selection import unsup_feature_selection
from model_cluster import clustering_method
from nlp import eda_dtale


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Supervised Learning Project","Unsupervised Learning Project","NLP Project"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.subheader("Dataset Preview")
        st.write(df.head())
        if st.checkbox("Numerical & Categorical Columns"):
            st.subheader("Column Types")
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            st.info(f"Number of Numerical Columns: {len(numerical_cols)}")
            st.write(f"Numerical Columns: {numerical_cols}")
            st.info(f"Number of Categorical Columns: {len(categorical_cols)}")
            st.write(f"Categorical Columns: {categorical_cols}")
            

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Supervised Learning Project":
    st.header("Regression_Classification_Prediction")
    if st.checkbox("Handling Missing Values"):
        data1=handle_missing_values(df)

    if st.checkbox("Handling ouliers"):
        data2=preprocess_outliers(data1)
        
    if st.checkbox("Target column"):
        column_list = data2.columns.tolist()
        target_column=st.selectbox("Select Your Target Column (Choose Dependent column for the target column)", column_list)
        # Drop the target column from X
        x = data2.drop(target_column, axis=1)
        # Extract the target column into y
        y = data2[target_column]

    if st.checkbox("Feature Scaling"):
        data3=perform_feature_scaling(x)

    if st.checkbox("Encoding treatment"):
        data4=preprocess_encoding(data3)

    if st.checkbox("Check Distribution & Transformation"):
        data5=distribution(data4)

    if st.checkbox("Feature selection"):
        data6 = feature_selection(data5, y)
        st.write("Final data before Model Training")
        st.write(data6)
        
    if st.checkbox("Model Training"):
        model_training(data6)
        
if choice == "Unsupervised Learning Project":
    st.header("Clustering_prediction")
    if st.checkbox("Handling Missing Values"):
        data21=handle_missing_values(df)

    if st.checkbox("Handling ouliers"):
        data22=preprocess_outliers(data21)
        
    if st.checkbox("Target column"):
        st.info("No Target,only Features for clustering ")
        column_list = data22.columns.tolist()
        target_column=st.selectbox("Select Your Target Column (Choose Dependent column for the target column)", column_list)
        # Drop the target column from X
        x = data22.drop(target_column, axis=1)
        # Extract the target column into y
        y = data22[target_column]

    if st.checkbox("Feature Scaling"):
        data23=perform_feature_scaling(x)

    if st.checkbox("Encoding treatment"):
        data24=preprocess_encoding(data23)

    if st.checkbox("Check Distribution & Transformation"):
        data25=distribution(data24)

    if st.checkbox("Feature selection"):
        data26 = unsup_feature_selection(data25, y)
        st.write("Final data before Clustering")
        st.write(data26)
        
    if st.checkbox("Clustering the Model"):
        clustering_method(data26)    
   

if choice == "NLP Project":
    st.header("Natural Language Processing")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        # Try different encodings until the file is read successfully
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                data = pd.read_csv(file, encoding=encoding)
                break  # Break the loop if the file is read successfully
            except UnicodeDecodeError:
                continue  # Try the next encoding if decoding fails
    st.write(data.head(5))
    st.write("Data shape :",data.shape)
    if st.checkbox("Perform EDA "):
        eda_dtale(data)
    
