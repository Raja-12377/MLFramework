import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_values(df):
    st.subheader("Missing Value Treatment")

    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_cols = missing_values[missing_values > 0].index.tolist()
    
    st.subheader("Missing Values")
    
    if not missing_cols:  # Check if there are any columns with missing values
        st.write("There are no missing values in the dataset")
    else:
        filtered_missing_values = missing_values[missing_values > 0]  # Filter out columns with zero missing values
        st.write(filtered_missing_values)  # Display missing values count for selected columns
        
    for i, col in enumerate(missing_cols):
        st.write(f"Column: {col}")
        if df[col].dtype == 'object':
            # Categorical column
            method = st.selectbox(f"Select Imputation Method for '{col}'", ["Select",
                    "Most Frequent Value", "Constant (e.g., 'missing')",
                    "Drop Entire Column"
                ],
                key=f"cat_{i}"  # Unique key for categorical selectbox
            )
            if method == "Most Frequent Value":
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                st.success("Missing value treatment is done.")

            elif method == "Constant (e.g., 'missing')":
                constant_value = st.text_input("Enter constant value:", value='missing')
                df[col].fillna(constant_value, inplace=True)
                st.success("Missing value treatment is done.")

            elif method == "Drop Entire Column":
                df.drop(col, axis=1, inplace=True)
                st.write(f"Column '{col}' dropped due to missing values.")
                st.success("Missing value treatment is done.")

        else:
            # Numeric column
            method = st.selectbox(f"Select Imputation Method for '{col}'", [
                    "Select", "Mean", "Median", "Mode", "KNN",
                    "Drop Entire Column", "Forward Fill", "Backward Fill", "Standard Deviation",
                    "Interpolation (Linear)", "Interpolation (Polynomial)", "Interpolation (Quadratic)"
                ],
                key=f"num_{i}"  # Unique key for numeric selectbox
            )
            if method == "Mean":
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                st.success("Missing value treatment is done.")

            elif method == "Median":
                imputer = SimpleImputer(strategy='median')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                st.success("Missing value treatment is done.")

            elif method == "Mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                st.success("Missing value treatment is done.")

            elif method == "KNN":
                k = st.slider(f"Select number of neighbors for KNN imputation for '{col}'", min_value=1, max_value=10, value=5)
                imputer = KNNImputer(n_neighbors=k)
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                st.success("Missing value treatment is done.")

            elif method == "Drop Entire Column":
                df.drop(col, axis=1, inplace=True)
                st.write(f"Column '{col}' dropped due to missing values.")
                st.success("Missing value treatment is done.")

            elif method == "Forward Fill":
                df[col].fillna(method='ffill', inplace=True)
                st.success("Missing value treatment is done.")

            elif method == "Backward Fill":
                df[col].fillna(method='bfill', inplace=True)
                st.success("Missing value treatment is done.")

            elif method == "Standard Deviation":
                std_dev = df[col].std()
                df[col].fillna(df[col].mean() + std_dev, inplace=True)
                st.success("Missing value treatment is done.")

            elif method.startswith("Interpolation"):
                interpolation_type = method.split(" ")[1].lower()
                df[col].interpolate(method=interpolation_type, inplace=True)
                st.success("Missing value treatment is done.")

        
    if st.checkbox("DataFrame after Imputation:"):
        m = df.isnull().sum()
        if m.all() == 0:
            st.write(m)
            st.success("There are no missing values in the dataset")
        else:
            st.write("Missing Values Count:")
            st.write(m)
    return df                    
# def main():
#     st.title("Missing Value Treatment App")

#     uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)

#         st.subheader("Original DataFrame")
#         st.write(data)

#         if st.checkbox("Handle Missing Values"):
#             handle_missing_values(data)

# if __name__ == '__main__':
#     main()
