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
from sklearn.utils import resample

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
.main {background-color: #f8fafc;}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
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

menu = st.sidebar.radio(
    "Menu",
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

    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='metric-card'><h4>Orders</h4><h2>{df['order_id'].nunique()}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h4>Customers</h4><h2>{df['customer_unique_id'].nunique()}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h4>Revenue</h4><h2>${df['payment_value'].sum():,.0f}</h2></div>", unsafe_allow_html=True)

    st.divider()

    category = st.selectbox("Category", df["product_category_name_english"].unique())
    df_filtered = df[df["product_category_name_english"] == category]

    fig = px.bar(df_filtered.groupby("product_id")["payment_value"].sum().head(10))
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

    fig = px.scatter(rfm, x="Frequency", y="Monetary", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm.groupby("cluster")[["Recency","Frequency","Monetary"]].mean())

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":
    st.title("🎯 Recommendation Product")

    from surprise import Dataset, Reader, SVD

    user_id = st.text_input("Customer ID")

    # =========================
    # LOAD + PREPARE DATA
    # =========================
    data_rec = df[["customer_unique_id", "product_id", "review_score"]].dropna()

    # Convert to string (important for Surprise)
    data_rec["customer_unique_id"] = data_rec["customer_unique_id"].astype(str)
    data_rec["product_id"] = data_rec["product_id"].astype(str)

    # =========================
    # TRAIN MODEL (cache để không train lại mỗi lần reload)
    # =========================
    @st.cache_resource
    def train_svd(data):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data, reader)
        trainset = dataset.build_full_trainset()

        model = SVD()
        model.fit(trainset)

        return model

    model = train_svd(data_rec)

    # =========================
    # RECOMMENDATION LOGIC
    # =========================
    if user_id:
        user_id = str(user_id)

        # ❗ Cold start
        if user_id not in data_rec["customer_unique_id"].unique():
            st.warning("Cold start → Recommend popular products")

            popular = (
                df.groupby("product_id")["review_score"]
                .count()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            st.dataframe(popular)
        else:
            # Products user already bought
            purchased = data_rec[
                data_rec["customer_unique_id"] == user_id
            ]["product_id"].unique()

            all_products = data_rec["product_id"].unique()

            predictions = []

            for product in all_products:
                if product not in purchased:
                    pred = model.predict(user_id, product)
                    predictions.append((product, pred.est))

            # Top 10
            top_10 = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

            rec_df = pd.DataFrame(top_10, columns=["product_id", "predicted_rating"])

            st.subheader("Top 10 Recommendations")
            st.dataframe(rec_df)

# =========================
# MARKET BASKET
# =========================
elif menu == "🛍️ Market Basket":

    st.title("🛍️ Market Basket Analysis")

    try:
        rules = pd.read_csv("data/rules.csv")

        col1, col2, col3 = st.columns(3)

        min_support = col1.slider("Min Support", 0.0, 1.0, 0.01)
        min_conf = col2.slider("Min Confidence", 0.0, 1.0, 0.1)
        min_lift = col3.slider("Min Lift", 0.0, 10.0, 1.0)

        filtered = rules[
            (rules["support"] >= min_support) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ]

        st.write(f"Rules: {len(filtered)}")

        fig = px.scatter(filtered, x="support", y="confidence", size="lift", color="lift")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filtered.head(20))

    except:
        st.error("rules.csv not found")

# =========================
# PREDICTION (FIX + PROB)
# =========================
elif menu == "🔮 Prediction":

    st.title("🔮 Prediction")

    col1, col2, col3 = st.columns(3)

    price = col1.number_input("Price", 0.0)
    freight = col2.number_input("Freight", 0.0)
    payment = col3.number_input("Payment", 0.0)

    payment_type = st.selectbox("Payment Type", df["payment_type"].unique())

    if st.button("Predict"):

        input_df = pd.DataFrame({
            "price":[price],
            "freight_value":[freight],
            "payment_value":[payment],
            "payment_type":[payment_type]
        })

        input_df = pd.get_dummies(input_df)

        if hasattr(pipeline, "feature_names_in_"):
            input_df = input_df.reindex(columns=pipeline.feature_names_in_, fill_value=0)

        pred = pipeline.predict(input_df)

        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(input_df)[0][1]

            st.success(f"""
            Prediction: {pred[0]}
            Confidence: {prob:.2f}
            """)
        else:
            st.success(f"Prediction: {pred[0]}")

# =========================
# ADMIN (FIX IMBALANCE)
# =========================
elif menu == "⚙️ Admin":

    st.title("⚙️ Admin")

    file = st.file_uploader("Upload CSV")

    if file:
        new_df = pd.read_csv(file)
        st.dataframe(new_df.head())

        if st.button("Retrain"):
            try:
                # BALANCE DATA
                df_major = new_df[new_df["review_score"] >= 4]
                df_minor = new_df[new_df["review_score"] < 4]

                df_minor_upsampled = resample(
                    df_minor,
                    replace=True,
                    n_samples=len(df_major),
                    random_state=42
                )

                balanced_df = pd.concat([df_major, df_minor_upsampled])

                X = balanced_df[["price","freight_value","payment_value","payment_type"]]
                X = pd.get_dummies(X)

                y = balanced_df["review_score"].apply(lambda x: 1 if x >= 4 else 0)

                pipeline.fit(X, y)
                joblib.dump(pipeline, "pipeline.pkl")

                st.success("Retrained with balanced data!")

                st.write("Label distribution:")
                st.write(y.value_counts())

            except Exception as e:
                st.error(e)