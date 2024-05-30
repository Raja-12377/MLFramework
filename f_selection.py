import streamlit as st
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

def feature_selection(data, y):
    st.subheader("Feature Selection")
    
    selection_methods = st.selectbox("Select Feature Selection Method", ["Select","Manual", "Automated (RFE)"])
    
    if "Manual" in selection_methods:
        selected_features = st.multiselect("Select Features", data.columns)
        selected_features_df = data[selected_features]
        
        # Combine selected features and y column
        final_df = pd.concat([selected_features_df, y], axis=1)
        return final_df
    
    if "Automated (RFE)" in selection_methods:
        k = st.number_input("Enter the number of features to select", min_value=1, max_value=len(data.columns), value=5, step=1)
        
        # Perform RFE with Gradient Boosting Classifier
        gbc = GradientBoostingClassifier()  # Initialize the classifier (use appropriate classifier)
        rfe_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
        rfe_selector.fit(data, y)
        
        # Get selected feature indices
        selected_feature_indices = rfe_selector.get_support(indices=True)
        
        # Get selected feature names
        selected_features = data.columns[selected_feature_indices].tolist()
        
        # Display selected feature names
        st.write(f"Selected Features (k={k}):")
        st.write(selected_features)
        
        selected_features_df = data[selected_features]
        
        # Combine selected features and y column
        final_df = pd.concat([selected_features_df, y], axis=1)
        return final_df