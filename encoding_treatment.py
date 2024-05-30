import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def preprocess_encoding(data):
    st.subheader("Encoding Treatment")
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if st.checkbox("View Categorical columns"):
        st.write(categorical_cols)
    for col in categorical_cols:
            encoding_method = st.selectbox(f"Select encoding method for '{col}'", ['Label Encoding', 'One-Hot Encoding'])
            if encoding_method == 'Label Encoding':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype('str'))
                st.write(f"Label Encoding for '{col}' is done.")
                categories = le.classes_  # Get the original categories
                st.write("Original Categories:", categories)
                st.write("Encoded Values:", data[col])
            elif encoding_method == 'One-Hot Encoding':
                data = pd.get_dummies(data, columns=[col], drop_first=True)
                st.write("OneHot Encoding is done")
    
    st.success("Encoding treatment is done.")
    st.write("Encoded Data",data)
    return data


