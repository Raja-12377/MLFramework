import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def distribution(data):
    for col in data.columns:
        st.write(f"Checking Distribution of Column: {col}")
        
        skewness = data[col].skew()
        kurtosis = data[col].kurt()
        # Define the checkbox keys before using them
        checkbox_key = f"checkbox_normal_{col}"
        checkbox_key2 = f"checkbox_not_normal_{col}"
        
        if -1 < skewness < 1 and -2 < kurtosis < 2:
            # Create the distribution plot
            if st.checkbox("View distribution in graph",key=checkbox_key):
                fig, ax = plt.subplots()
                sns.distplot(data[col], ax=ax)
                ax.set_title(f"Distribution of Column: {col}")
                # Display the plot using st.pyplot()
                st.pyplot(fig)
            st.info(f"Skewness: {skewness} (within acceptable range for normal distribution)")
            st.info(f"Kurtosis: {kurtosis} (within acceptable range for normal distribution)")
            st.success(f"Column: {col} is normally distributed.")
        else:
            if st.checkbox("View distribution in graph",key=checkbox_key2):
                fig, ax = plt.subplots()
                sns.distplot(data[col], ax=ax)
                ax.set_title(f"Distribution of Column: {col}")
                # Display the plot using st.pyplot()
                st.pyplot(fig)
            st.info(f"Skewness: {skewness} (not within acceptable range for normal distribution)")
            st.info(f"Kurtosis: {kurtosis} (not within acceptable range for normal distribution)")
            st.warning(f"Column: {col} is not normally distributed.")
            
            transformation_method = st.selectbox(f"Select Transformation Method for '{col}'", 
                                                 ['Sqrt Transformation', 'Log Transformation', 'Cube Transformation'])
            
            if transformation_method == 'Sqrt Transformation':
                data[col] = np.where(data[col] < data[col].quantile(0.05), np.sqrt(data[col]), data[col])
                data[col] = np.where(data[col] > data[col].quantile(0.95), np.sqrt(data[col]), data[col])
            elif transformation_method == 'Log Transformation':
                data[col] = np.where(data[col] < data[col].quantile(0.05), np.log(data[col]), data[col])
                data[col] = np.where(data[col] > data[col].quantile(0.95), np.log(data[col]), data[col])
            elif transformation_method == 'Cube Transformation':
                data[col] = np.where(data[col] < data[col].quantile(0.05), np.power(data[col], 3), data[col])
                data[col] = np.where(data[col] > data[col].quantile(0.95), np.power(data[col], 3), data[col])
        
    return data
