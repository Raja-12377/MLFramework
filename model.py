import streamlit as st
import pandas as pd
import numpy as np
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
import numba
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Binarizer, StandardScaler, Normalizer
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.datasets import make_friedman1
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.impute import SimpleImputer


def model_training(data):
    scaled_data=data
    
    # Allow user to select multiple columns
    column_list = data.columns.tolist()
    target_col = st.selectbox("Please Select Your Target Column Once again for Training", column_list)
    st.write("Data shape : ",scaled_data.shape)
    if st.checkbox("Check NaN values in data"):
        nan_count = scaled_data.isnull().sum().sum()
        st.write("Total NaN values in the data:", nan_count)
        
        if nan_count >= 1:
            handle_method = st.selectbox("Select a method to handle NaN values:", ["Select","Drop rows with NaN", "Impute with mean"])
            
            if handle_method == "Drop rows with NaN":
                if st.checkbox("Drop rows with NaN"):
                    scaled_data.dropna(inplace=True)
                    nan_count_after = data.isnull().sum().sum()
                    st.write("Total NaN values in the data after dropping:", nan_count_after)
                    st.write("Data shape after dropping rows with NaN:", data.shape)
                    st.write(scaled_data)
            
            elif handle_method == "Impute with mean":
                if st.checkbox("Impute NaN values with mean"):
                    imputer = SimpleImputer(strategy='mean')
                    scaled_data_imputed = imputer.fit_transform(data)
                    scaled_data = pd.DataFrame(scaled_data_imputed, columns=data.columns)
                    nan_count_after = data.isnull().sum().sum()
                    st.write("Total NaN values in the data after imputation:", nan_count_after)
                    st.write("Data shape after imputation:", data.shape)
                    st.write(scaled_data)
    
    if st.checkbox("Perform Model training"):
        
        if target_col:
            st.write(f"Your selected target column: {target_col}")

            # Check if target column is numerical or categorical
            if scaled_data[target_col].dtype in ['int64', 'float64']:
                st.write(f"Target column '{target_col}' is numerical")

                # Determine if numerical column is continuous or discrete
                unique_count = scaled_data[target_col].nunique()
                if unique_count / len(scaled_data[target_col]) < 0.05:
                    st.write(f"Target column '{target_col}' is discrete")
                    target_type = 'discrete'
                    st.write(f"Target column '{target_col}' is discrete , we have to choose First Decision Tree regression  Algorithm ")
                    st.header("Regression-discrete Model Selector")            
                    button_clicked = st.checkbox("generate algorithm")
                    if button_clicked:
                        perform_regression_discrete(scaled_data,target_col)

                else:
                    target_type = 'continuous'
                    st.write(f"Target column '{target_col}' is continuous , we have to choose First LinearRegression Algorithm ")
                    st.header("Regression-Continous Model Selector")
                    button_clicked = st.checkbox("Generate algorithm")
                    if button_clicked:
                        perform_regression_continous(scaled_data,target_col)

            else:
                st.write(f"Target column '{target_col}' is categorical")
                unique_values = scaled_data[target_col].unique()
                is_ordinal = True
                try:
                    sorted_values = sorted(unique_values, key=float)
                    is_ordinal = all(x == sorted_values[i] for i, x in enumerate(unique_values))
                except ValueError:
                    is_ordinal = False

                if is_ordinal:
                    st.write(f"Target column '{target_col}' is ordinal.we have to choose first Descision Tree classifier")
                    st.header("Classification-ordinal Model Selector")
                    if st.checkbox("Check Balanced or Imbalanced data"):
                        X = scaled_data.drop(columns=[target_col])
                        y = scaled_data[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        class_counts = Counter(y_train)
                        if max(class_counts.values()) / min(class_counts.values()) < 2:
                            st.info("Data's are Balanced...")
                            X_train_resampled, y_train_resampled = X_train, y_train
                            if st.checkbox("Generate Algorithm"):
                                Bperform_classification_ordinal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test)
                        else:
                            st.info("Data's are Imbalanced...applying SMOTE technique to balance the data")
                            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
                            if st.checkbox("Generate Algorithm"):
                              IBperform_classification_ordinal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test)
                    
                else:
                    st.write(f"Target column '{target_col}' is nominal.we have to choose Logistic Regression")
                    st.header("Classification-Nominal Model Selector")
                    if st.checkbox("Check Balanced or Imbalanced data"):
                        X = scaled_data.drop(columns=[target_col])
                        y = scaled_data[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        class_counts = Counter(y_train)
                        if max(class_counts.values()) / min(class_counts.values()) < 2:
                            st.info("Data's are Balanced...")
                            X_train_resampled, y_train_resampled = X_train, y_train
                            if st.checkbox("Generate Algorithm"):
                                Bperform_classification_nominal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test)
                        else:
                            st.info("Data's are Imbalanced...applying SMOTE technique to balance the data")
                            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
                            if st.checkbox("Generate Algorithm"):
                              IBperform_classification_nominal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test)

def apply_smote(X_train, y_train, random_state=27):
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    st.write("Class distribution before SMOTE:", Counter(y_train))
    st.write("Class distribution after SMOTE:", Counter(y_train_resampled))
    
    return X_train_resampled, y_train_resampled

def calculate_adjusted_r2_score(r2, n, p):
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

def preprocess_user_input(input_data, X_train_resampled):

    # Handle missing values (fill with appropriate values, e.g., median from X_train)
    input_data.fillna(X_train_resampled.median(), inplace=True)  # Example: Fill missing values with median
    
    # Encode categorical variables using the same encoder used on X_train
    for col in input_data.columns:
        if X_train_resampled[col].dtype == 'object':  # Check if column is categorical
            encoder = LabelEncoder()  # Use LabelEncoder for simplicity (replace with OneHotEncoder if needed)
            encoder.fit(X_train_resampled[col])  # Fit encoder on X_train column
            input_data[col] = encoder.transform(input_data[col])  # Transform user input column
    
    # Reindex columns to match X_train columns and fill missing columns with zeros
    input_data = input_data.reindex(columns=X_train_resampled.columns, fill_value=0)
    
    return input_data

def linear_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = LinearRegression()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    evaluation_metrics = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }
    
    return evaluation_metrics, model

def polynomial_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_resampled)
    X_test_poly = poly_features.transform(X_test)
    
    return linear_regression(X_train_poly, X_test_poly, y_train_resampled, y_test)

def lasso_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = Lasso()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    evaluation_metrics = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }
    
    return evaluation_metrics, model

def ridge_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = Ridge()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    evaluation_metrics = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }
    
    return evaluation_metrics, model

def elastic_net_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = ElasticNet()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    evaluation_metrics = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }
    
    return evaluation_metrics, model

def perform_regression_continous(scaled_data, target_col):
    X = scaled_data.drop(columns=[target_col])
    y = scaled_data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Imbalanced or balanced if needs
    X_train_resampled, y_train_resampled = X_train, y_train
    algorithms = [
        ("Linear Regression", linear_regression),
        ("Polynomial Regression", polynomial_regression),
        ("Lasso Regression", lasso_regression),
        ("Ridge Regression", ridge_regression),
        ("Elastic Net Regression", elastic_net_regression)
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")

        # Evaluate the algorithm and get the evaluation metrics and trained model
        evaluation_metrics, model = algorithm_func(X_train_resampled, X_test, y_train_resampled, y_test)

        r2 = evaluation_metrics["r2"]
        adjusted_r2 = evaluation_metrics["adjusted_r2"]
        mae = evaluation_metrics["mae"]
        mse = evaluation_metrics["mse"]
        rmse = evaluation_metrics["rmse"]

        if r2 >= 0.8 or adjusted_r2 >= 0.8:
            st.write(f"{algorithm_name} is good. R-squared (R2) =  {r2:.2f} , Adjusted R-squared =  {adjusted_r2:.2f} Mean Absolute Error (MAE): {mae:.2f} , Mean Squared Error (MSE): {mse:.2f} , Root Mean Squared Error (RMSE): {rmse:.2f} ")
             # Allow user to make predictions with this regression model
            st.subheader("Prediction")
            predict_button = st.checkbox("Predict with selected model")
            if predict_button:
                selected_columns = st.multiselect("Select columns for prediction", X.columns)
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input
                
                    if st.button("Make Prediction"):
                        input_data = pd.DataFrame([user_inputs])
                        # Ensure input_data columns match X_train columns and order
                   
                        input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                        prediction = model.predict(input_data_processed[selected_columns])
                        # Display the prediction result
                        st.subheader("Prediction Result")
                        st.success(f"Predicted Output: {prediction[0]}")
            
            break
            
        else:
            st.write(f" R-squared (R2) =  {r2:.2f} , Adjusted R-squared (R2) =  {adjusted_r2:.2f}  are low for {algorithm_name} Mean Absolute Error (MAE): {mae:.2f} , Mean Squared Error (MSE): {mse:.2f} , Root Mean Squared Error (RMSE): {rmse:.2f} . Choosing next algorithm.")
   

def decision_tree_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_train_resampled.shape[1]
    adjusted_r2 = calculate_adjusted_r2_score(r2, n, p)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, adjusted_r2, mae, mse, rmse, model

def random_forest_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = RandomForestRegressor()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_train_resampled.shape[1]
    adjusted_r2 = calculate_adjusted_r2_score(r2, n, p)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, adjusted_r2, mae, mse, rmse, model

def xgboost_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = XGBRegressor()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_train_resampled.shape[1]
    adjusted_r2 = calculate_adjusted_r2_score(r2, n, p)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, adjusted_r2, mae, mse, rmse, model

def sv_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    model = SVR()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    n = X_test.shape[0]
    p = X_train_resampled.shape[1]
    adjusted_r2 = calculate_adjusted_r2_score(r2, n, p)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, adjusted_r2, mae, mse, rmse, model


def perform_regression_discrete(scaled_data, target_col):
    X = scaled_data.drop(columns=[target_col])
    y = scaled_data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Imbalanced or balanced if needs
    X_train_resampled, y_train_resampled = X_train, y_train

    algorithms = [
        ("Decision Tree Regression", decision_tree_regression),
        ("Random Forest Regression", random_forest_regression),
        ("XGBoost Regression", xgboost_regression),
        ("SV Regression", sv_regression)
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")

        # Evaluate the algorithm and get the evaluation metrics and trained model
        r2, adjusted_r2, mae, mse, rmse, model = algorithm_func(X_train_resampled, X_test, y_train_resampled, y_test)
        
        if r2 >= 0.8 or adjusted_r2 >= 0.8:
            st.write(f"{algorithm_name} is good. R-squared (R2) =  {r2:.2f} , Adjusted R-squared =  {adjusted_r2:.2f} Mean Absolute Error (MAE): {mae:.2f} , Mean Squared Error (MSE): {mse:.2f} , Root Mean Squared Error (RMSE): {rmse:.2f} ")
            # Allow user to make predictions with this regression model
            st.subheader("Prediction")
            predict_button = st.checkbox("Predict with selected model")

            if predict_button:
                selected_columns = st.multiselect("Select columns for prediction", scaled_data.columns.drop(target_col))
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input
                
                if st.button("Make Prediction"):
                    input_data = pd.DataFrame([user_inputs])
                    input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                    prediction = model.predict(input_data_processed)
                    
                    # Display the prediction result
                    st.subheader("Prediction Result")
                    st.success(f"Predicted Output: {prediction[0]}")
            break
        else:
            st.write(f" R-squared (R2) =  {r2:.2f} , Adjusted R-squared (R2) =  {adjusted_r2:.2f}  are low for {algorithm_name} Mean Absolute Error (MAE): {mae:.2f} , Mean Squared Error (MSE): {mse:.2f} , Root Mean Squared Error (RMSE): {rmse:.2f} . Choosing next algorithm.")

def decision_tree_classifier(X_train_resampled, X_test, y_train_resampled, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return evaluation_metrics, model

def random_forest_classifier(X_train_resampled, X_test, y_train_resampled, y_test):
    model = RandomForestClassifier()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Return classification evaluation metrics and trained model
    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return evaluation_metrics, model

def xgboost_classifier(X_train_resampled, X_test, y_train_resampled, y_test):
    model = XGBClassifier()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Return classification evaluation metrics and trained model
    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return evaluation_metrics, model

def logistic_regression(X_train_resampled, X_test, y_train_resampled, y_test):
    # Instantiate the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Return classification evaluation metrics and trained model
    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return evaluation_metrics, model
 

def Bperform_classification_ordinal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test):
    algorithms = [
    ("Decision Tree Classifier", decision_tree_classifier),
    ("Random Forest Classifier", random_forest_classifier),
    ("XGBoost Classifier", xgboost_classifier),
    # Add more classifiers as needed
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")
        
        # Evaluate the algorithm and get the accuracy and trained model
        evaluation_metrics, model = algorithm_func(X_train_resampled,X_test,y_train_resampled,y_test,scaled_data)
        accuracy = evaluation_metrics["accuracy"]
        if accuracy >= 0.8:
            st.write(f"{algorithm_name} is good. Accuracy =  {accuracy:.2f}")
            st.subheader("Prediction")
            predict_button = st.checkbox("Predict with selected classifier")

            if predict_button:
                selected_columns = st.multiselect("Select columns for prediction", scaled_data.columns.drop(target_col))
                
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input

                    if st.button("Make Prediction"):
                        input_data = pd.DataFrame([user_inputs])
                        input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                        prediction = model.predict(input_data_processed)
                        st.subheader("Prediction Result")
                        st.success(f"Predicted Output: {prediction[0]}")
            
            break
        else:
            st.write(f"Accuracy =  {accuracy:.2f}. Accuracy is low for {algorithm_name}. Choosing next algorithm.")

def IBperform_classification_ordinal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test):
    algorithms = [
    ("Decision Tree Classifier", decision_tree_classifier),
    ("Random Forest Classifier", random_forest_classifier),
    ("XGBoost Classifier", xgboost_classifier),
    # Add more classifiers as needed
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")
        
        # Evaluate the algorithm and get the evaluation metrics, model, and scaled data
        evaluation_metrics, model = algorithm_func(X_train_resampled, X_test, y_train_resampled, y_test)
        
        precision = evaluation_metrics["precision"]
        recall = evaluation_metrics["recall"]
        f1_score = evaluation_metrics["f1_score"]
        
        if precision >= 0.8 or recall >= 0.8 or f1_score >= 0.8:
            st.write(f"{algorithm_name} is good. Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1_score:.2f}")
            
            # Allow user to make predictions with this classifier
            st.subheader("Prediction")
            predict_button = st.checkbox("Predict with selected classifier")

            if predict_button:
                selected_columns = st.multiselect("Select columns for prediction", scaled_data.columns.drop(target_col))
                
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input

                    if st.button("Make Prediction"):
                        input_data = pd.DataFrame([user_inputs])
                        input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                        prediction = model.predict(input_data_processed)
                        st.subheader("Prediction Result")
                        st.success(f"Predicted Output: {prediction[0]}")
                        
            break
        else:
            st.write(f"{algorithm_name} has precision = {precision:.2f}, recall = {recall:.2f}, f1_score = {f1_score:.2f} values are low, choosing next algorithm")


def Bperform_classification_nominal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test):
    algorithms = [
        ("Logistic Regression", logistic_regression),
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")
        
        # Evaluate the algorithm and get the accuracy and trained model
        evaluation_metrics, model = algorithm_func(X_train_resampled, X_test, y_train_resampled, y_test)
        accuracy = evaluation_metrics["accuracy"]
        if accuracy >= 0.8:
            st.success(f"{algorithm_name} is good. Accuracy =  {accuracy:.2f}")
            
            # Allow user to select columns for prediction
            if st.checkbox("Go Prediction"):
                # target_col = label  # replace with the actual target column name
                selected_columns = st.multiselect("Select columns for prediction", scaled_data.columns.drop(target_col))
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input
                    
                    if st.button("Predict"):
                        # Create input_data DataFrame with user inputs
                        input_data = pd.DataFrame([user_inputs])
                        
                        # Preprocess user inputs to match model expectations
                        input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                        
                        # Make prediction using the trained model
                        predictions = model.predict(input_data_processed[selected_columns])
                        
                        # Display the prediction result
                        st.subheader("Prediction Result")
                        st.success(f"Predicted Output: {predictions[0]}")
            
            break
        else:
            st.write(f"Accuracy =  {accuracy:.2f}. Accuracy is low for {algorithm_name}. ")

def IBperform_classification_nominal(X,y,scaled_data,target_col,X_train_resampled,X_test,y_train_resampled,y_test):
    algorithms = [
        ("Logistic Regression", logistic_regression),
    ]

    for algorithm_name, algorithm_func in algorithms:
        st.info(f"Performing {algorithm_name}...")
        
        # Evaluate the algorithm and get the accuracy and trained model
        evaluation_metrics, model = algorithm_func(X_train_resampled, X_test, y_train_resampled, y_test)
        precision = evaluation_metrics["precision"]
        recall = evaluation_metrics["recall"]
        f1_score = evaluation_metrics["f1_score"]
        if precision >= 0.8 or recall >= 0.8 or f1_score >= 0.8:
            st.success(f"{algorithm_name} is good. precision =  {precision:.2f} , recall =  {recall:.2f} ,  f1_score =  {precision:.2f}")
            
            # Allow user to select columns for prediction
            if st.checkbox("Go Prediction for another values"):
                selected_columns = st.multiselect("Select columns for prediction", scaled_data.columns.drop(target_col))
                
                if selected_columns:
                    st.subheader("Enter Values for Prediction")
                    user_inputs = {}
                    
                    for col in selected_columns:
                        user_input = st.text_input(f"Enter value for '{col}'", key=col)
                        user_inputs[col] = user_input
                    
                    if st.button("Make Prediction"):
                        # Create input_data DataFrame with user inputs
                        input_data = pd.DataFrame([user_inputs])
                        
                        # Preprocess user inputs to match model expectations
                        input_data_processed = preprocess_user_input(input_data, X_train_resampled)
                        
                        # Make prediction using the trained model
                        predictions = model.predict(input_data_processed[selected_columns])
                        
                        # Display the prediction result
                        st.subheader("Prediction Result")
                        st.success(f"Predicted Output: {predictions[0]}")
            
            break
        else:
            st.write(f"{algorithm_name} of precision =  {precision:.2f} ,  recall =  {recall:.2f} , f1_score =  {precision:.2f} values are low")




    
    