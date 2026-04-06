import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛒 Customer Segmentation using K-Means")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error("Need at least 2 numeric columns!")
    else:
        numeric_df = numeric_df.fillna(numeric_df.mean())

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        k = st.slider("Select number of clusters", 2, 10, 4)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        df['Cluster'] = clusters

        summary = df.groupby('Cluster').mean(numeric_only=True)

        label_column = None

        for col in numeric_df.columns:
            if "spend" in col.lower() or "amount" in col.lower():
                label_column = col
                break

        if label_column is None:
            label_column = numeric_df.columns[0]  # fallback

        cluster_order = summary[label_column].sort_values().index

        labels_map = {}
        names = [
        "Inactive / One-Time Buyers",
        "Occasional Shoppers",
        "Regular Customers",
        "High-Value Loyal Customers",
        "Big Spenders",
        "VIP Customers",
        "Discount Seekers",
        "At-Risk Customers",
        "New Customers",
        "Frequent Browsers"
        ]

        for i, cluster_id in enumerate(cluster_order):
            if i < len(names):
                labels_map[cluster_id] = names[i]
            else:
                labels_map[cluster_id] = f"Group {i}"

        df['Segment'] = df['Cluster'].map(labels_map)

        st.subheader("📊 Clustered Data")
        st.dataframe(df.head())

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots(figsize=(4, 2))
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
        ax.set_title("Customer Segments Visualization")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.pyplot(fig)

        st.subheader("📈 Cluster Summary")
        st.dataframe(summary)

        st.subheader("📊 Segment Distribution")
        st.write(df['Segment'].value_counts())

        st.subheader("🧠 Insights")
        for seg in df['Segment'].unique():
            st.write(f"### {seg}")
            st.write(df[df['Segment'] == seg].describe())

        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Dataset",
            data=csv,
            file_name='segmented_customers.csv',
            mime='text/csv',
        )