import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# ===================== CONFIG =====================
st.set_page_config(page_title="E-commerce Analytics", layout="wide")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data_small.csv")

# ===================== HEAD ========================
df = load_data()

# ===================== SIDEBAR =====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose Page",
    ["Dashboard", "Clustering", "Recommendation", "Market Basket", "Prediction", "Admin"] 
) 

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 MENU")

menu = st.sidebar.radio(
    "Chọn chức năng",
    [
        "📊 Dashboard",
        "👥 Segmentation",
        "🎯 Recommendation",
        "🛍️ Market Basket",
        "🔮 Prediction",
        "⚙️ Admin"
    ]

)

# ===================== DASHBOARD =====================
if page == "Dashboard":
    st.title("📊 E-commerce Dashboard")

    col1, col2, col3 = st.columns(3)

    total_orders = len(df)

    total_revenue = df["Monetary"].sum() if "Monetary" in df.columns else 0

    total_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else 0

    col1.metric("🛒 Orders", total_orders)
    col2.metric("💰 Revenue", f"{total_revenue:,.0f}")
    col3.metric("👤 Customers", total_customers)

    st.subheader("📈 Revenue Distribution")
    if "Monetary" in df.columns:
        fig = px.histogram(df, x="Monetary", nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Monetary' not found!")

# ===================== CLUSTERING =====================
elif page == "Clustering":
    st.title("🧠 Customer Segmentation (RFM)")

    if all(col in df.columns for col in ["Recency", "Frequency", "Monetary"]):
        X = df[["Recency", "Frequency", "Monetary"]]

        kmeans = KMeans(n_clusters=4, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        st.subheader("Cluster Summary")
        st.dataframe(df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean())

        fig = px.scatter(
            df,
            x="Recency",
            y="Monetary",
            color="Cluster",
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Missing RFM columns!")

# ===================== PREDICTION =====================
elif page == "Prediction":
    st.title("🤖 Customer Prediction")

    recency = st.number_input("Recency", 0)
    frequency = st.number_input("Frequency", 0)
    monetary = st.number_input("Monetary", 0)

    if st.button("Predict"):
        if monetary > 1000:
            st.success("🌟 VIP Customer")
        elif monetary > 500:
            st.info("👍 Loyal Customer")
        else:
            st.warning("⚠️ Normal Customer")

# ===================== RECOMMENDATION =====================
elif page == "Recommendation":
    st.title("🛒 Product Recommendation")

    customer_id = st.text_input("Enter Customer ID")

    if st.button("Recommend"):
        st.subheader("Top Recommended Products")
        st.write(["Product A", "Product B", "Product C"])

# ===================== FP-GROWTH =====================
elif page == "Market Basket":
    st.title("📈 Association Rules")

    try:
        rules = pd.read_csv("data/rules.csv")
        st.dataframe(rules.head(20))
    except:
        st.warning("rules.csv not found!")

# ===================== ADMIN =====================
elif page == "Admin":
    st.title("⚙️ Admin Panel - Retrain Model")

    st.subheader("📤 Upload New Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(new_df.head())

        if st.button("🔄 Retrain Clustering Model"):
            if all(col in new_df.columns for col in ["Recency", "Frequency", "Monetary"]):
                X = new_df[["Recency", "Frequency", "Monetary"]]

                kmeans = KMeans(n_clusters=4, random_state=42)
                new_df["Cluster"] = kmeans.fit_predict(X)

                st.success("Model retrained successfully!")

                st.subheader("Cluster Summary")
                st.dataframe(new_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean())

                fig = px.scatter(
                    new_df,
                    x="Recency",
                    y="Monetary",
                    color="Cluster",
                    title="New Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Dataset must contain RFM columns!")

# ===================== FOOTER =====================
st.sidebar.markdown("---")
st.sidebar.info("Big Data ML Project 🚀")