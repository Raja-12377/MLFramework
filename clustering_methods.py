elif sup_unsup_method == "clustering method":
        clustering_method(scaled_data)

def clustering_method(scaled_data):
    # Ask the user to select feature columns
    selected_columns = st.multiselect("Select feature columns:", scaled_data.columns)
    
    # Create the feature matrix X with selected columns
    X = scaled_data[selected_columns]
    
    clustering_method = st.selectbox("Select a Clustering Method:", ["select", "Centroid Cluster: K-Means", "Density-Based Cluster: DBSCAN", "Hierarchical Cluster: Agglomerative"])
    
    if clustering_method == "Centroid Cluster: K-Means":
        # Implement K-Means clustering method
        k = st.number_input("Enter the number of clusters (K) for K-Means:", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        st.write("Cluster Centers (K-Means):")
        st.write(cluster_centers)
        
        st.write("Cluster Labels (K-Means):")
        st.write(cluster_labels)
        # Ask the user if they want to perform silhouette analysis
        perform_silhouette_analysis = st.checkbox("Perform Silhouette Analysis for K-Means")
        if perform_silhouette_analysis:
            # Perform silhouette analysis
            silhouette_avg = silhouette_score(X, cluster_labels)
            st.write(f"Average Silhouette Score: {silhouette_avg:.2f}")
            if silhouette_avg < 0.5:
                st.write("Silhouette score is near 0. Proceed with caution for prediction.")
                # Ask the user for data to predict the cluster
                st.write("Enter data to predict the cluster for K-Means:")
                user_input = st.text_area("Input data (comma-separated values)")
        
                data = pd.DataFrame([map(float, user_input.split(','))])
                predicted_cluster = kmeans.predict(data)
        
                st.write("Predicted Cluster (K-Means):")
                st.write(predicted_cluster)
            else:
                st.write("Silhouette score is near 1. Clustering may not be suitable.")
        
    elif clustering_method == "Density-Based Cluster: DBSCAN":
        # Implement DBSCAN clustering method
        eps = st.number_input("Enter the maximum distance between samples for DBSCAN:", value=0.5)
        min_samples = st.number_input("Enter the number of samples in a neighborhood for DBSCAN:", value=5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels_dbscan = dbscan.fit_predict(X)
        
        st.write("Cluster Labels (DBSCAN):")
        st.write(cluster_labels_dbscan)
        
        # Ask the user for data to predict the cluster
        st.write("Enter data to predict the cluster for DBSCAN:")
        user_input_dbscan = st.text_area("Input data (comma-separated values)")
        
        data_dbscan = pd.DataFrame([map(float, user_input_dbscan.split(','))])
        predicted_cluster_dbscan = dbscan.fit_predict(data_dbscan)
        
        st.write("Predicted Cluster (DBSCAN):")
        st.write(predicted_cluster_dbscan)
        
    elif clustering_method == "Hierarchical Cluster: Agglomerative":
        # Implement Agglomerative clustering method
        n_clusters = st.number_input("Enter the number of clusters for Agglomerative:", min_value=2, value=3)
        
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels_agg = agglomerative.fit_predict(X)
        
        st.write("Cluster Labels (Agglomerative):")
        st.write(cluster_labels_agg)
        
        # Ask the user for data to predict the cluster
        st.write("Enter data to predict the cluster for Agglomerative:")
        user_input_agg = st.text_area("Input data (comma-separated values)")
        
        data_agg = pd.DataFrame([map(float, user_input_agg.split(','))])
        predicted_cluster_agg = agglomerative.fit_predict(data_agg)
        
        st.write("Predicted Cluster (Agglomerative):")
        st.write(predicted_cluster_agg)


#dimension reduction - PCA for large data sets 
from sklearn.decomposition import PCA

# Assuming 'scaled_data' is your scaled input data

# Initialize PCA with the desired number of components
pca = PCA(n_components=2)  # You can adjust the number of components as needed

# Fit PCA to the scaled data and transform the data
X_pca = pca.fit_transform(scaled_data)

# Print the explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Print the transformed data after PCA
print("Transformed Data after PCA:")
print(X_pca)

# Based on the information from these sources, it is appropriate to use PCA before K-means clustering because PCA reduces dimensionality, captures strong patterns, and helps in visualizing data in a lower-dimensional space, making clustering more effective and interpretable. PCA simplifies the data by emphasizing variation, making it easier for clustering algorithms like K-means to identify patterns and groupings within the data.

#Market based analysis - Recommentaion engine -assoctive learing -apropri

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Assuming 'scaled_data' is your scaled input data

# Convert the scaled data to a binary format suitable for Apriori
binary_data = scaled_data.applymap(lambda x: 1 if x > 0 else 0)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(binary_data, min_support=0.1, use_colnames=True)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print the association rules with support, confidence, and lift
print("\nAssociation Rules:")
for index, rule in rules.iterrows():
    pre = ', '.join(list(rule.antecedents))
    post = ', '.join(list(rule.consequents))
    print(f"{pre} -> {post}")
    print(f"Support: {rule['support']:.3f}")
    print(f"Confidence: {rule['confidence']:.3f}")
    print(f"Lift: {rule['lift']:.3f}")
    print()