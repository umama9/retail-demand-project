import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Retail Demand Dashboard")

st.title("Retail Demand & Pricing Intelligence")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_excel(r"Online Retail.xlsx")
    df = df.dropna(subset=["Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df["Description"].str.isupper()]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    return df

df = load_data()
 
# ---------- Product Lists ----------
product_sales = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False)

top_products = product_sales.head(20).index.tolist()
low_products = product_sales.tail(20).index.tolist()

product_group = st.selectbox("Choose Product Category",["Top Selling Products", "Low Selling Products"])

if product_group == "Top Selling Products":
    product = st.selectbox("Select Product", top_products)
else:
    product = st.selectbox("Select Product", low_products)

product_data = df[df["Description"] == product]

# ---------- Dashboard Metrics ----------
col1, col2, col3 = st.columns(3)

col1.metric("Total Units Sold", int(product_data["Quantity"].sum()))
col2.metric("Average Price", round(product_data["UnitPrice"].mean(),2))
col3.metric("Transactions", product_data["InvoiceNo"].nunique())

st.divider()

# ---------- Demand Curve ----------
st.subheader("Price vs Demand (Demand Curve)")

price_demand = product_data.groupby("UnitPrice")["Quantity"].sum().reset_index()
price_demand = price_demand.sort_values("UnitPrice")

fig, ax = plt.subplots()

ax.plot(price_demand["UnitPrice"], price_demand["Quantity"], marker="o")

ax.set_xlabel("Price")
ax.set_ylabel("Quantity sold")
ax.set_title("Demand Curve")

st.pyplot(fig)

# ---------- Monthly Sales ----------
st.subheader("Monthly Sales Trend")

monthly_sales = product_data.groupby("Month")["Quantity"].sum().reset_index()

monthly_sales["Month"] = monthly_sales["Month"].astype(str)

fig2, ax2 = plt.subplots()

ax2.plot(monthly_sales["Month"], monthly_sales["Quantity"], marker="o")

ax2.set_xlabel("Month")
ax2.set_ylabel("Units Sold")
ax2.set_title("Monthly Sales Trend")

plt.xticks(rotation=45)

st.pyplot(fig2)

# ---------- Prediction Model ----------
st.subheader("Sales Prediction")

monthly_sales["MonthIndex"] = range(len(monthly_sales))

if len(monthly_sales) < 2:
    st.warning("Not enough data to make prediction")
else:
    X = monthly_sales[["MonthIndex"]]
    y = monthly_sales["Quantity"]
    model = LinearRegression()
    model.fit(X, y)
    next_month = np.array([[len(monthly_sales)]])
    prediction = model.predict(next_month)[0]
    prediction = max(0, prediction)
    st.success(f"Predicted units sold next month: {int(prediction)}")

# ---------- Price Optimization ----------
st.subheader("Recommended Maximum Price")

price_qty = product_data.groupby("UnitPrice")["Quantity"].sum().reset_index()

X_price = price_qty[["UnitPrice"]]
y_qty = price_qty["Quantity"]

model2 = LinearRegression()
model2.fit(X_price,y_qty)

current_price = product_data["UnitPrice"].mean()

price_range = np.linspace(current_price, current_price*1.5,50)

predicted_demand = model2.predict(price_range.reshape(-1,1))

acceptable_prices = price_range[predicted_demand > predicted_demand.max()*0.7]

if len(acceptable_prices) > 0:
    recommended_price = max(acceptable_prices)
else:
    recommended_price = current_price

st.info(f"Maximum price before strong demand drop: {recommended_price:.2f}")

# ---------- Raw Data ----------
if st.checkbox("Show Raw Data"):
    st.dataframe(product_data.head(50))
