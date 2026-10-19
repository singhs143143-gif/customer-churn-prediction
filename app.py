import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_data.csv"
MODEL_PATH = BASE_DIR / "best_churn_pipeline.pkl"

st.title("Customer Churn Prediction")
st.caption("Explore customer trends and estimate churn risk using your trained ML model.")

# ---------------------------------------------------------
# Load data and model
# ---------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place customer_data.csv in the same folder as app.py.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run main.py first so best_churn_pipeline.pkl is created.")
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

if "churn" not in df.columns:
    st.error("The dataset must contain a 'churn' column.")
    st.stop()

# Use the same dataset-derived threshold used by the training data.
high_balance_threshold = float(df["balance"].quantile(0.75))

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Explore filters")
countries = sorted(df["country"].dropna().unique().tolist())
genders = sorted(df["gender"].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries
)
selected_genders = st.sidebar.multiselect(
    "Gender",
    genders,
    default=genders
)

min_age = int(df["age"].min())
max_age = int(df["age"].max())
age_range = st.sidebar.slider(
    "Age range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

filtered = df[
    df["country"].isin(selected_countries)
    & df["gender"].isin(selected_genders)
    & df["age"].between(age_range[0], age_range[1])
].copy()

# ---------------------------------------------------------
# Navigation tabs
# ---------------------------------------------------------
overview_tab, explore_tab, predict_tab = st.tabs(
    ["Overview", "Explore customers", "Predict churn"]
)

with overview_tab:
    st.subheader("Customer overview")
    st.caption("The metrics and charts below update when you change the sidebar filters.")

    total_customers = len(filtered)
    churned = int((filtered["churn"] == 1).sum())
    retained = int((filtered["churn"] == 0).sum())
    churn_rate = (churned / total_customers * 100) if total_customers else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Churned", f"{churned:,}")
    c3.metric("Retained", f"{retained:,}")
    c4.metric("Churn rate", f"{churn_rate:.1f}%")

    if filtered.empty:
        st.warning("No customers match these filters. Change the filters in the sidebar.")
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("**Churn distribution**")
            churn_counts = (
                filtered["churn"]
                .map({0: "Stayed", 1: "Churned"})
                .value_counts()
                .reindex(["Stayed", "Churned"], fill_value=0)
            )
            st.bar_chart(churn_counts)

        with right:
            st.markdown("**Average churn rate by country**")
            country_rate = (
                filtered.groupby("country", observed=False)["churn"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(country_rate)

        st.markdown("**Churn rate by age group**")
        filtered["age_group_view"] = pd.cut(
            filtered["age"],
            bins=[17, 24, 34, 44, 54, 64, 120],
            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        )
        age_rate = (
            filtered.groupby("age_group_view", observed=False)["churn"]
            .mean()
            .mul(100)
            .dropna()
        )
        st.line_chart(age_rate)

with explore_tab:
    st.subheader("Explore customer records")
    st.write("Use the filters in the sidebar to narrow the records shown here.")

    if filtered.empty:
        st.warning("No records match the current filters.")
    else:
        show_churned_only = st.checkbox("Show only customers who churned", value=False)
        table_data = filtered.copy()

        if show_churned_only:
            table_data = table_data[table_data["churn"] == 1]

        display_columns = [
            col for col in [
                "customer_id", "credit_score", "country", "gender", "age",
                "tenure", "balance", "products_number", "active_member",
                "estimated_salary", "churn"
            ] if col in table_data.columns
        ]
        st.dataframe(
            table_data[display_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = table_data[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered customer data (CSV)",
            data=csv_data,
            file_name="filtered_customer_data.csv",
            mime="text/csv"
        )

with predict_tab:
    st.subheader("Predict churn for one customer")
    st.write("Enter customer details and click **Predict** to get the model's estimate.")

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.slider("Credit score", 350, 850, 650)
            country = st.selectbox("Country", countries if countries else ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", genders if genders else ["Male", "Female"])
            age = st.slider("Age", 18, 92, 35)

        with col2:
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Account balance", min_value=0.0, value=50000.0, step=1000.0)
            products_number = st.selectbox("Number of products", [1, 2, 3, 4])

        with col3:
            credit_card_label = st.selectbox("Has credit card?", ["Yes", "No"])
            active_member_label = st.selectbox("Active member?", ["Yes", "No"])
            estimated_salary = st.number_input(
                "Estimated salary", min_value=0.0, value=75000.0, step=1000.0
            )

        predict_clicked = st.form_submit_button(
            "Predict churn",
            use_container_width=True
        )

    if predict_clicked:
        # Feature engineering must match main.py.
        balance_per_product = balance / products_number
        salary_balance_ratio = np.nan if balance == 0 else estimated_salary / balance

        if age < 25:
            age_group = "18-24"
        elif age < 35:
            age_group = "25-34"
        elif age < 45:
            age_group = "35-44"
        elif age < 55:
            age_group = "45-54"
        elif age < 65:
            age_group = "55-64"
        else:
            age_group = "65+"

        if tenure <= 2:
            tenure_bucket = "1-2"
        elif tenure <= 5:
            tenure_bucket = "3-5"
        else:
            tenure_bucket = "6-10"

        high_balance = int(balance > high_balance_threshold)

        customer = pd.DataFrame([{
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": int(credit_card_label == "Yes"),
            "active_member": int(active_member_label == "Yes"),
            "estimated_salary": estimated_salary,
            "balance_per_product": balance_per_product,
            "salary_balance_ratio": salary_balance_ratio,
            "age_group": age_group,
            "tenure_bucket": tenure_bucket,
            "high_balance": high_balance
        }])

        try:
            prediction = int(model.predict(customer)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                # The training target uses 0 = stayed and 1 = churned.
                classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
                if 1 in classes:
                    probability = float(probabilities[classes.index(1)])

            st.divider()
            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("Predicted outcome: Customer may churn")
                    st.write("Consider reviewing this customer's engagement and service needs.")
                else:
                    st.success("Predicted outcome: Customer is likely to stay")
                    st.write("Continue regular customer engagement and monitoring.")

            with probability_col:
                if probability is not None:
                    st.metric("Estimated churn probability", f"{probability:.1%}")
                    st.progress(max(0.0, min(1.0, probability)))
                    st.caption("Probability is a model estimate, not a guarantee.")

            with st.expander("View features sent to the model"):
                st.dataframe(customer, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info(
                "Check that feature engineering and category labels in app.py "
                "match those used when training the model in main.py."
            )

st.divider()
st.caption("Customer Churn Prediction • Machine-learning decision support")