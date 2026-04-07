# =========================
# IMPORT
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
# STYLE
# =========================
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f8fafc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    color: white;
}

/* KPI Card */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

/* Text */
h1, h2, h3 {
    color: #111827;
}

/* Button */
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    font-weight: 600;
}

/* Input */
.stTextInput input, .stNumberInput input {
    border-radius: 10px;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Slider */
.stSlider > div {
    color: #6366f1;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA & PIPELINE
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data_small.csv")

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("pipeline.pkl")
    except:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

df = load_data()
pipeline = load_pipeline()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚀 E-commerce App")
st.sidebar.markdown("### Analytics Dashboard")
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

    col1, col2, col3 = st.columns(3, gap="large")

    col1.markdown(f"""
    <div class='metric-card'>
    <h4>🛒 Orders</h4>
    <h2>{df['order_id'].nunique()}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class='metric-card'>
    <h4>👤 Customers</h4>
    <h2>{df['customer_unique_id'].nunique()}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class='metric-card'>
    <h4>💰 Revenue</h4>
    <h2>${df['payment_value'].sum():,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    category_filter = st.selectbox("Chọn Category", df["product_category_name_english"].unique())
    df_filtered = df[df["product_category_name_english"] == category_filter]

    fig = px.bar(
        df_filtered.groupby("product_id")["payment_value"].sum().head(10),
        title="Top Products",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":

    st.title("👥 Customer Segmentation")

    rfm = df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": "max",
        "order_id": "count",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer", "Recency", "Frequency", "Monetary"]
    rfm["Recency"] = (pd.to_datetime("today") - pd.to_datetime(rfm["Recency"])).dt.days

    scaler = MinMaxScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    k = st.slider("Clusters", 2, 8, 4)

    model = KMeans(n_clusters=k, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="cluster",
        color_continuous_scale="viridis"
    )
    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm.groupby("cluster")[["Recency","Frequency","Monetary"]].mean())

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":

    st.title("🎯 Product Recommendation")

    user_id = st.text_input("Nhập Customer ID")

    if user_id:

        data = df[["customer_unique_id","product_id","review_score"]].dropna()

        if user_id not in data["customer_unique_id"].astype(str).values:
            st.warning("Cold start → Recommend popular")

            popular = df.groupby("product_id")["review_score"].count().sort_values(ascending=False).head(10)
            st.dataframe(popular)

        else:
            rec = df.groupby("product_id")["review_score"].mean().sort_values(ascending=False).head(10)
            st.dataframe(rec)

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
            color="lift",
            color_continuous_scale="plasma"
        )
        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rules.head(20))

    except:
        st.error("Chưa có rules.csv")

# =========================
# PREDICTION
# =========================
elif menu == "🔮 Prediction":

    st.title("🔮 Predict Customer Satisfaction")

    col1, col2, col3 = st.columns(3)

    price = col1.number_input("Price", 0.0)
    freight = col2.number_input("Freight", 0.0)
    payment = col3.number_input("Payment", 0.0)

    payment_type = st.selectbox("Payment Type", df["payment_type"].unique())

    if st.button("Predict"):
        with st.spinner("🔄 Predicting..."):

            input_df = pd.DataFrame({
                "price": [price],
                "freight_value": [freight],
                "payment_value": [payment],
                "payment_type": [payment_type]
            })

            pred = pipeline.predict(input_df)

            if hasattr(pipeline, "predict_proba"):
                prob = pipeline.predict_proba(input_df)[0][1]
                st.success(f"Prediction: {pred[0]} | Confidence: {prob:.2f}")
            else:
                st.success(f"Prediction: {pred[0]}")

# =========================
# ADMIN
# =========================
elif menu == "⚙️ Admin":

    st.title("⚙️ Admin Panel")

    file = st.file_uploader("Upload CSV")

    if file:
        new_df = pd.read_csv(file)
        st.dataframe(new_df.head())

        if st.button("Retrain"):

            try:
                X = new_df[["price","freight_value","payment_value","payment_type"]]
                X = pd.get_dummies(X)

                y = new_df["review_score"].apply(lambda x: 1 if x >= 4 else 0)

                pipeline.fit(X, y)
                joblib.dump(pipeline, "pipeline.pkl")

                st.success("Model updated!")

            except Exception as e:
                st.error(e)