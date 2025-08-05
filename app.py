
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load pre-trained models
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))                  # KMeans clustering model for customer segmentation
similarity_df = pickle.load(open("product_similarity.pkl", "rb"))           # Cosine similarity matrix for product recommendations

# Streamlit app title
st.title("🛒 Shopper Spectrum")

#  Create tabs for two functionalities: product recommendations and customer segmentation
tab1, tab2 = st.tabs(["Product Recommendations", "Customer Segmentation"])

# ==============================================
#  TAB 1: PRODUCT RECOMMENDATIONS
# ==============================================
with tab1:
    st.subheader("🔍Get Similar Products")
    product = st.text_input("Enter Product Code (e.g., 84029E)")  # Input product code

    if st.button("Recommend"):
        if product in similarity_df.columns:
            # Get top 5 similar products based on cosine similarity
            similar_items = similarity_df[product].sort_values(ascending=False)[1:6]
            st.success("Top 5 Recommendations:")
            for i, item in enumerate(similar_items.index):
                st.write(f"{i+1}. Product Code: {item}")
        else:
            st.error("Product not found.")
            
            

# ==============================================
#  TAB 2: CUSTOMER SEGMENTATION
# ==============================================
with tab2:
    st.subheader("📊 Customer Segment Prediction")
    
    # Accept Recency, Frequency, Monetary values as user input
    r = st.number_input("Recency (in days)", min_value=0)
    f = st.number_input("Frequency (number of purchases)", min_value=0)
    m = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):
        # Standardize the input before predicting
        scaler = StandardScaler()
        scaled_input = scaler.fit_transform([[r, f, m]])

        # Predict the cluster using the trained KMeans model
        cluster = kmeans_model.predict(scaled_input)[0]

        # Map cluster to human-readable segment labels
        segments = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }

        # Show predicted segment
        st.info(f"Predicted Segment: **{segments.get(cluster, 'Unknown')}**")
    