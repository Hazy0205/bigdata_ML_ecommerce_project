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
# STYLE
# =========================
st.markdown("""
<style>
.metric-card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
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

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":

    st.title("📊 E-commerce Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='metric-card'>Orders<br><h2>{df['order_id'].nunique()}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'>Customers<br><h2>{df['customer_unique_id'].nunique()}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'>Revenue<br><h2>${df['payment_value'].sum():,.0f}</h2></div>", unsafe_allow_html=True)

    st.divider()

    category_filter = st.selectbox("Chọn Category", df["product_category_name_english"].dropna().unique())
    df_filtered = df[df["product_category_name_english"] == category_filter]

    st.plotly_chart(
        px.bar(
            df_filtered.groupby("product_id")["payment_value"].sum().sort_values(ascending=False).head(10),
            title="Top Products"
        ),
        use_container_width=True
    )

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

    st.plotly_chart(
        px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"),
        use_container_width=True
    )

    st.dataframe(rfm.groupby("cluster")[["Recency","Frequency","Monetary"]].mean())

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":

    st.title("🎯 Recommendation Product")

    # SEARCH
    keyword = st.text_input("🔍 Search product_id")

    if keyword:
        result = df[df["product_id"].astype(str).str.contains(keyword)]
        st.dataframe(result[["product_id","payment_value"]].head(10))

    st.divider()

    user_id = st.text_input("Customer ID")

    data_rec = df[["customer_unique_id","product_id","review_score"]].dropna()

    if len(data_rec) == 0:
        st.error("No data for recommendation")
        st.stop()

    data_rec = data_rec.sample(n=min(20000, len(data_rec)), random_state=42)

    top_products = data_rec["product_id"].value_counts().head(100).index
    data_rec = data_rec[data_rec["product_id"].isin(top_products)]

    @st.cache_data
    def create_pivot(data):
        return data.pivot_table(
            index="customer_unique_id",
            columns="product_id",
            values="review_score"
        ).fillna(0)

    pivot = create_pivot(data_rec)

    if user_id:

        user_id = str(user_id)

        if user_id not in pivot.index:
            st.warning("Cold start → Recommend popular")

            popular = (
                df.groupby("product_id")["review_score"]
                .count()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            st.dataframe(popular)

        else:
            user_vector = pivot.loc[user_id]

            similarity = pivot.dot(user_vector)
            similarity = similarity.drop(index=user_id)

            similar_users = similarity.sort_values(ascending=False).head(5).index

            rec = pivot.loc[similar_users].mean().sort_values(ascending=False)

            purchased = user_vector[user_vector > 0].index
            rec = rec.drop(labels=purchased, errors="ignore").head(10)

            rec_df = rec.reset_index()
            rec_df.columns = ["product_id","score"]

            st.success("Top Recommendations")
            st.dataframe(rec_df)

# =========================
# MARKET BASKET
# =========================
elif menu == "🛍️ Market Basket":

    st.title("🛍️ Market Basket")

    try:
        rules = pd.read_csv("data/rules.csv")

        st.plotly_chart(
            px.scatter(rules, x="support", y="confidence", size="lift", color="lift"),
            use_container_width=True
        )

        st.dataframe(rules.head(20))

    except:
        st.error("Chưa có datarules.csv")

# =========================
# PREDICTION 
# =========================
elif menu == "🔮 Prediction":

    st.title("🔮 Predict Customer Satisfaction")

    # PREPARE DATA
    df_pred = df[[
        "price",
        "freight_value",
        "payment_value",
        "payment_type",
        "review_score"
    ]].dropna()

    df_pred["target"] = df_pred["review_score"].apply(lambda x: 1 if x >= 4 else 0)

    # ENCODE
    df_pred = pd.get_dummies(df_pred, columns=["payment_type"], drop_first=True)

    X = df_pred.drop(columns=["review_score", "target"])
    y = df_pred["target"]

    # TRAIN MODEL
    from sklearn.ensemble import RandomForestClassifier

    @st.cache_resource
    def train_model(X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    model = train_model(X, y)

    st.success("Model trained successfully!")

    # INPUT
    col1, col2, col3 = st.columns(3)

    price = col1.number_input("Price", 0.0)
    freight = col2.number_input("Freight", 0.0)
    payment = col3.number_input("Payment", 0.0)

    payment_type = st.selectbox("Payment Type", df["payment_type"].dropna().unique())

    # BUILD INPUT
    input_df = pd.DataFrame({
        "price": [price],
        "freight_value": [freight],
        "payment_value": [payment],
        "payment_type": [payment_type]
    })

    input_df = pd.get_dummies(input_df)

    # ALIGN COLUMNS
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # PREDICT
    if st.button("Predict"):

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"😊 Good Review (>=4) | Confidence: {prob:.2f}")
        else:
            st.error(f"😞 Bad Review (<4) | Confidence: {prob:.2f}")

# =========================
# ADMIN (RETRAIN MODEL)
# =========================
elif menu == "⚙️ Admin":

    st.title("⚙️ Admin Panel - Retrain Model")

    file = st.file_uploader("Upload CSV")

    if file:
        new_df = pd.read_csv(file)
        st.dataframe(new_df.head())

        required_cols = ["price","freight_value","payment_value","payment_type","review_score"]

        if not all(col in new_df.columns for col in required_cols):
            st.error("❌ File thiếu cột cần thiết")
            st.stop()

        # PREPROCESS
        df_train = new_df[required_cols].dropna()
        df_train["target"] = df_train["review_score"].apply(lambda x: 1 if x >= 4 else 0)

        df_train = pd.get_dummies(df_train, columns=["payment_type"], drop_first=True)

        X = df_train.drop(columns=["review_score","target"])
        y = df_train["target"]

        from sklearn.ensemble import RandomForestClassifier

        # SESSION STATE (lưu model)
        if "model" not in st.session_state:
            st.session_state.model = None

        if st.button("🚀 Retrain Model"):

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            st.session_state.model = model
            st.session_state.columns = X.columns

            st.success("✅ Model retrained successfully!")

        # TEST MODEL
        if st.session_state.get("model") is not None:

            st.subheader("🔮 Test Model")

            col1, col2, col3 = st.columns(3)

            price = col1.number_input("Price", 0.0)
            freight = col2.number_input("Freight", 0.0)
            payment = col3.number_input("Payment", 0.0)

            payment_type = st.selectbox("Payment Type", new_df["payment_type"].dropna().unique())

            input_df = pd.DataFrame({
                "price": [price],
                "freight_value": [freight],
                "payment_value": [payment],
                "payment_type": [payment_type]
            })

            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=st.session_state.columns, fill_value=0)

            if st.button("Predict (Admin)"):

                pred = st.session_state.model.predict(input_df)[0]
                prob = st.session_state.model.predict_proba(input_df)[0][1]

                if pred == 1:
                    st.success(f"😊 Good | Confidence: {prob:.2f}")
                else:
                    st.error(f"😞 Bad | Confidence: {prob:.2f}")