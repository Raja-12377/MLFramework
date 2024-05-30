import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def preprocess_outliers(data):
    st.subheader("Outliers Treatment")
    
    for col in data.columns:
        st.write(f"Column: {col}")
        
        # Check if column data is numeric
        if data[col].dtype not in ['int64', 'float64']:
            st.write(f"Skipping '{col}' (non-numeric column)")
            continue
        
        # Display boxplot before outlier treatment
        checkbox_key_before = f"outliers_checkbox_before_{col}"
        if st.checkbox("View Boxplot Before Outliers Treatment", key=checkbox_key_before):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[col])
            plt.title(f'Boxplot of "{col}" Before Outlier Treatment')
            plt.xlabel(col)
            plt.ylabel('Values')
            st.pyplot()

        lower_limit = data[col].mean() - 3 * data[col].std()
        upper_limit = data[col].mean() + 3 * data[col].std()
        
        # Apply outlier treatment based on user selection
        replace_method = st.selectbox(f"Select Method to Replace Outliers for '{col}'", 
                                      ['Trimming Approach', 'Imputation', 'Capping Approach', 'No Treatment'])
        if replace_method == 'Trimming Approach':
            data = data[(data[col] >= lower_limit) & (data[col] <= upper_limit)]
        elif replace_method == 'Imputation':
            imputation_method = st.selectbox(f"Select Imputation Method for '{col}'", 
                                             ['Constant Value', 'Mean', 'Median', 'Mode', 'Standard Deviation'])
            if imputation_method == 'Constant Value':
                value = st.number_input(f"Enter Constant Value to Replace Outliers for '{col}'", value=0.0)
                data[col] = np.where((data[col] < lower_limit) | (data[col] > upper_limit), value, data[col])
            elif imputation_method in ['Mean', 'Median', 'Mode']:
                method_func = getattr(data[col], imputation_method.lower())
                replace_value = method_func()
                data[col] = np.where((data[col] < lower_limit) | (data[col] > upper_limit), replace_value, data[col])
            elif imputation_method == 'Standard Deviation':
                std_dev = data[col].std()
                mean_value = data[col].mean()
                data[col] = np.where((data[col] < lower_limit) | (data[col] > upper_limit), mean_value + std_dev, data[col])

        elif replace_method == 'Capping Approach':
            capping_method = st.selectbox(f"Select Capping Method for '{col}'", 
                                           ['select','Z-score Method', 'IQR Method'])
            if capping_method == 'Z-score Method':
                z_score = st.number_input("Enter Z-score Threshold", value=2.0)
                data[col] = np.where((data[col] - data[col].mean()).abs() > z_score * data[col].std(), np.nan, data[col])
                
            elif capping_method == 'IQR Method':
                data[col] = np.where((data[col] < lower_limit) | (data[col] > upper_limit), np.nan, data[col])
        
        # Display boxplot after outlier treatment
        checkbox_key_after = f"outliers_checkbox_after_{col}"
        if st.checkbox("View Boxplot After Outliers Treatment", key=checkbox_key_after):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[col])
            plt.title(f'Boxplot of "{col}" After Outlier Treatment')
            plt.xlabel(col)
            plt.ylabel('Values')
            st.pyplot()
            
    st.success("Outliers treatment is done.")
    return data

# def main():
#     st.title("Outliers Treatment App")

#     # Upload dataset
#     st.subheader("Upload Your Dataset")
#     uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

#     if uploaded_file is not None:
#         st.subheader("Preview of Uploaded Dataset")
#         data = pd.read_csv(uploaded_file)
#         st.write(data.head())

#         # Check for outliers and apply treatment
#         if st.checkbox("Check Outliers and Apply Treatment"):
#             treated_data = preprocess_outliers(data)
#             st.subheader("Treated Dataset")
#             st.write(treated_data.head())

# if __name__ == "__main__":
#     main()