# =========================
# IMPORT
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="E-commerce Analytics",
    layout="wide",
    page_icon="🚀"
)

# =========================
# STYLE (UI XỊN)
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-card h2 {
    font-size: 28px;
    margin: 10px 0;
}

.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data_small.csv")

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 🚀 E-commerce App")
st.sidebar.markdown("---")

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

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":

    st.title("📊 E-commerce Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class='metric-card'>
    <h4>🛒 Orders</h4>
    <h2>{df.shape[0]}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class='metric-card'>
    <h4>👤 Customers</h4>
    <h2>{df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class='metric-card'>
    <h4>💰 Revenue</h4>
    <h2>{df['Monetary'].sum() if 'Monetary' in df.columns else 0:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if "Monetary" in df.columns:
        fig = px.histogram(df, x="Monetary", nbins=50)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":

    st.title("👥 Customer Segmentation")

    if all(col in df.columns for col in ["Recency", "Frequency", "Monetary"]):

        X = df[["Recency", "Frequency", "Monetary"]]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Clusters", 2, 8, 4)

        model = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = model.fit_predict(X_scaled)

        fig = px.scatter(
            df,
            x="Frequency",
            y="Monetary",
            color="Cluster"
        )
        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean())

    else:
        st.error("Missing RFM columns")

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":

    st.title("🎯 Product Recommendation")

    customer_id = st.text_input("Nhập Customer ID")

    if customer_id:
        st.markdown("### 🔥 Top Recommendations")

        rec = df.groupby("product_id")["review_score"].mean().sort_values(ascending=False).head(10)

        st.dataframe(rec, use_container_width=True)

# =========================
# MARKET BASKET
# =========================
elif menu == "🛍️ Market Basket":

    st.title("🛍️ Market Basket")

    try:
        rules = pd.read_csv("data/rules.csv")

        fig = px.scatter(
            rules,
            x="support",
            y="confidence",
            size="lift",
            color="lift"
        )
        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rules.head(20))

    except:
        st.warning("rules.csv not found")

# =========================
# PREDICTION (FAKE MODEL)
# =========================
elif menu == "🔮 Prediction":

    st.title("🔮 Customer Prediction")

    recency = st.number_input("Recency", 0)
    frequency = st.number_input("Frequency", 0)
    monetary = st.number_input("Monetary", 0)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            if monetary > 1000:
                st.success("🌟 VIP Customer")
            elif monetary > 500:
                st.info("👍 Loyal Customer")
            else:
                st.warning("⚠️ Normal Customer")

# =========================
# ADMIN
# =========================
elif menu == "⚙️ Admin":

    st.title("⚙️ Admin Panel")

    st.info("Upload dataset để retrain clustering")

    file = st.file_uploader("Upload CSV")

    if file:
        new_df = pd.read_csv(file)
        st.dataframe(new_df.head())

        if st.button("Retrain"):
            if all(col in new_df.columns for col in ["Recency", "Frequency", "Monetary"]):
                X = new_df[["Recency","Frequency","Monetary"]]

                model = KMeans(n_clusters=4, random_state=42)
                new_df["Cluster"] = model.fit_predict(X)

                st.success("Retrain completed!")

                st.dataframe(new_df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean())
            else:
                st.error("Missing RFM columns")