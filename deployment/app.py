from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "reports" / "PJMW_EDA_dataset.csv"

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Power Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------------------------------------
# Load model and feature information
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "xgboost_model.pkl")
    features = joblib.load(BASE_DIR / "feature_info.pkl")
    return model, features


model, features = load_model()

# ------------------------------------------------------------
# Load historical dataset
# ------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Datetime" not in df.columns:
        raise ValueError("The dataset must contain a Datetime column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df


data = load_data()

# ------------------------------------------------------------
# Target column
# ------------------------------------------------------------

if "PJMW_MW_Clean" in data.columns:
    target = "PJMW_MW_Clean"
elif "PJMW_MW" in data.columns:
    target = "PJMW_MW"
else:
    st.error("Power demand column not found in the dataset.")
    st.stop()

# ------------------------------------------------------------
# Page content
# ------------------------------------------------------------

st.title("⚡ Power Demand Forecasting")
st.write("30-Day Power Demand Forecast using XGBoost")

st.success("Model and historical data loaded successfully.")

st.subheader("Historical Power Demand")

st.line_chart(
    data.set_index("Datetime")[target].tail(24 * 14)
)

# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------

def create_features(df):
    df = df.copy()

    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["Month"] = df["Datetime"].dt.month
    df["DayOfYear"] = df["Datetime"].dt.dayofyear
    df["WeekOfYear"] = (
        df["Datetime"].dt.isocalendar().week.astype(int)
    )
    df["Quarter"] = df["Datetime"].dt.quarter
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    def nth_weekday(year, month, weekday, n):
        first_day = pd.Timestamp(year, month, 1)
        offset = (weekday - first_day.weekday()) % 7
        return first_day + pd.Timedelta(
            days=offset + (n - 1) * 7
        )

    def last_weekday(year, month, weekday):
        last_day = (
            pd.Timestamp(year, month, 1)
            + pd.offsets.MonthEnd(1)
        )
        offset = (last_day.weekday() - weekday) % 7
        return last_day - pd.Timedelta(days=offset)

    holiday_dates = set()

    for year in range(
        df["Datetime"].dt.year.min(),
        df["Datetime"].dt.year.max() + 1
    ):
        holiday_dates.add(pd.Timestamp(year, 1, 1))
        holiday_dates.add(nth_weekday(year, 1, 0, 3))
        holiday_dates.add(nth_weekday(year, 2, 0, 3))
        holiday_dates.add(last_weekday(year, 5, 0))
        holiday_dates.add(pd.Timestamp(year, 7, 4))
        holiday_dates.add(nth_weekday(year, 9, 0, 1))

        thanksgiving = nth_weekday(year, 11, 3, 4)
        holiday_dates.add(thanksgiving)
        holiday_dates.add(thanksgiving + pd.Timedelta(days=1))

        holiday_dates.add(pd.Timestamp(year, 12, 24))
        holiday_dates.add(pd.Timestamp(year, 12, 25))
        holiday_dates.add(pd.Timestamp(year, 12, 31))

    df["Is_Holiday"] = (
        df["Datetime"].dt.normalize()
        .isin(holiday_dates)
        .astype(int)
    )

    df["Lag_1"] = df[target].shift(1)
    df["Lag_24"] = df[target].shift(24)
    df["Lag_168"] = df[target].shift(168)

    df["Rolling_Mean_24"] = (
        df[target].shift(1).rolling(24, min_periods=24).mean()
    )
    df["Rolling_Std_24"] = (
        df[target].shift(1).rolling(24, min_periods=24).std()
    )
    df["Rolling_Mean_168"] = (
        df[target].shift(1).rolling(168, min_periods=168).mean()
    )
    df["Rolling_Std_168"] = (
        df[target].shift(1).rolling(168, min_periods=168).std()
    )

    return df


# ------------------------------------------------------------
# Generate 30-day forecast
# ------------------------------------------------------------

st.subheader("30-Day Power Demand Forecast")

if st.button("Generate Forecast"):
    history = data.copy()
    future_predictions = []
    forecast_hours = 30 * 24

    progress = st.progress(0)

    for step in range(forecast_hours):
        next_datetime = (
            history["Datetime"].iloc[-1] + pd.Timedelta(hours=1)
        )

        new_row = pd.DataFrame({
            "Datetime": [next_datetime],
            target: [np.nan]
        })

        temp = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        temp = create_features(temp)
        X_future = temp[features].iloc[[-1]]

        prediction = float(model.predict(X_future)[0])

        new_row[target] = prediction

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        future_predictions.append({
            "Datetime": next_datetime,
            "Forecast_MW": prediction
        })

        if (step + 1) % 24 == 0:
            progress.progress((step + 1) / forecast_hours)

    progress.empty()

    forecast_df = pd.DataFrame(future_predictions)

    st.success("30-day forecast generated successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Demand",
        f"{forecast_df['Forecast_MW'].mean():,.2f} MW"
    )
    col2.metric(
        "Peak Demand",
        f"{forecast_df['Forecast_MW'].max():,.2f} MW"
    )
    col3.metric(
        "Minimum Demand",
        f"{forecast_df['Forecast_MW'].min():,.2f} MW"
    )

    st.subheader("Forecast Trend")
    st.line_chart(
        forecast_df.set_index("Datetime")["Forecast_MW"]
    )

    st.subheader("Forecast Results")
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="30_day_power_demand_forecast.csv",
        mime="text/csv"
    )
