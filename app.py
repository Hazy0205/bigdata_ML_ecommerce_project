import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-commerce Analytics", layout="wide")

# =========================
# LOAD DATA
# =========================
def load_data(file=None):
    if file:
        return pd.read_csv(file)
    return pd.read_csv("cleaned_data_small.csv")

# =========================
# RFM FUNCTION (CLEAN + SAFE)
# =========================
def create_rfm(data):
    rfm = data.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": "max",
        "order_id": "count",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_id","Recency","Frequency","Monetary"]

    # Convert Recency
    rfm["Recency"] = (pd.to_datetime("today") - pd.to_datetime(rfm["Recency"], errors='coerce')).dt.days

    # Convert numeric
    for col in ["Recency","Frequency","Monetary"]:
        rfm[col] = pd.to_numeric(rfm[col], errors="coerce")

    # Clean data
    rfm = rfm.replace([np.inf, -np.inf], np.nan)
    rfm = rfm.dropna(subset=["Recency","Frequency","Monetary"])

    return rfm

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Menu",
    [
        "📊 Dashboard",
        "👥 Segmentation",
        "🎯 Recommendation",
        "🛍️ Market Basket",
        "🔮 Prediction",
        "⚙️ Admin",
    ],
)

# =========================
# DATA
# =========================
df = load_data()

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", df["order_id"].nunique())
    col2.metric("Customers", df["customer_unique_id"].nunique())
    col3.metric("Revenue", f"${df['payment_value'].sum():,.0f}")

    st.divider()

    # Charts
    top_cat = df.groupby("product_category_name_english")["payment_value"].sum().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(top_cat, title="Top Categories"), use_container_width=True)

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    time_df = df.groupby(df["order_purchase_timestamp"].dt.date)["order_id"].count()
    st.plotly_chart(px.line(time_df, title="Orders Over Time"), use_container_width=True)

    # Clustering
    st.subheader("Customer Clustering Preview")
    rfm = create_rfm(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=4, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"), use_container_width=True)

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":
    st.title("👥 Customer Segmentation")

    file = st.file_uploader("Upload CSV", type=["csv"])
    data = load_data(file) if file else df

    rfm = create_rfm(data)

    k = st.slider("Clusters", 2, 8, 4)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=k, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"), use_container_width=True)

    st.subheader("Cluster Profile")
    # Only numeric columns for mean
    numeric_cols = ["Recency","Frequency","Monetary"]
    st.dataframe(rfm.groupby("cluster")[numeric_cols].mean())

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
        rules = pd.read_csv("rules.csv")

        min_lift = st.slider("Min Lift", 0.0, 10.0, 1.0)
        filtered = rules[rules["lift"] >= min_lift]

        st.dataframe(filtered.sort_values("lift", ascending=False).head(20))
        st.plotly_chart(px.scatter(filtered, x="support", y="confidence", size="lift"))

    except:
        st.warning("Run FP-Growth first")

# =========================
# PREDICTION
# =========================
elif menu == "🔮 Prediction":
    st.title("🔮 Prediction")

    price = st.number_input("Price", min_value=0.0)
    freight = st.number_input("Freight", min_value=0.0)
    payment = st.number_input("Payment", min_value=0.0)

    if st.button("Predict"):
        try:
            model = joblib.load("classifier.pkl")
            pred = model.predict([[price, freight, payment]])
            st.success(f"Prediction: {pred[0]}")
        except:
            st.error("Train model in Admin tab first")

# =========================
# ADMIN
# =========================
elif menu == "⚙️ Admin":
    st.title("⚙️ Admin Panel")

    file = st.file_uploader("Upload new dataset", type=["csv"])

    if file:
        new_df = pd.read_csv(file)
        st.success("Uploaded!")
        st.dataframe(new_df.head())

        if st.button("Retrain Model"):
            try:
                # Clean features
                X = new_df[["price","freight_value","payment_value"]]
                y = new_df["review_score"]

                # Convert numeric
                X = X.apply(pd.to_numeric, errors="coerce")
                y = pd.to_numeric(y, errors="coerce")

                # Combine and drop NaN together
                data_clean = pd.concat([X, y], axis=1).dropna()
                X = data_clean[["price","freight_value","payment_value"]]
                y = data_clean["review_score"]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                pred = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, pred))
                mae = mean_absolute_error(y, pred)

                joblib.dump(model, "classifier.pkl")

                st.success("Model retrained")
                st.write("RMSE:", rmse)
                st.write("MAE:", mae)

            except Exception as e:
                st.error(f"Error: {e}")



