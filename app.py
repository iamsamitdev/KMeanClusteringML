import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model from the file
kmeans_model = joblib.load('bmx_kmean.joblib')

st.title('K-Means Clustering')

# Upload the dataset and save as csv
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Select the columns for clustering
    clustering_columns = ['bmxleg', 'bmxwaist']

    # Drop rows with missing values in selected columns
    data = data.dropna(subset=clustering_columns)

    # Check if there are enough data points for clustering
    if data.shape[0] > kmeans_model.n_clusters:
        # Perform clustering
        cluster_labels = kmeans_model.predict(data[clustering_columns])

        # Add cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        st.write(data)

        # Plot the clusters
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Scatterplot")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='bmxleg', y='bmxwaist', hue='cluster')
        plt.title('KMeans Clustering')
        st.pyplot()
    else:
        st.write("Not enough data points for clustering")
