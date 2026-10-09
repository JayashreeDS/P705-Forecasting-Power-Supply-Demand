import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Power Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------------------------------------
# Load trained XGBoost model
# ------------------------------------------------------------

model = joblib.load("xgboost_model.pkl")

# ------------------------------------------------------------
# Load feature information
# ------------------------------------------------------------

features = joblib.load("feature_info.pkl")

# ------------------------------------------------------------
# Load historical dataset
# ------------------------------------------------------------

data = pd.read_csv("../reports/PJMW_EDA_dataset.csv")

data["Datetime"] = pd.to_datetime(data["Datetime"])

data = data.sort_values("Datetime").reset_index(drop=True)

# ------------------------------------------------------------
# Target column
# ------------------------------------------------------------

target = "PJMW_MW_Clean"

st.title("⚡ Power Demand Forecasting")

st.write(
    "30-Day Power Demand Forecast using XGBoost"
)

st.success("Model and data loaded successfully.")



# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------
def create_features(df):
    df = df.copy()

    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["Month"] = df["Datetime"].dt.month
    df["DayOfYear"] = df["Datetime"].dt.dayofyear
    df["WeekOfYear"] = df["Datetime"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Datetime"].dt.quarter

    df["Is_Weekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    def nth_weekday(year, month, weekday, n):
        first_day = pd.Timestamp(year, month, 1)
        days_to_weekday = (
            weekday - first_day.weekday()
        ) % 7

        return first_day + pd.Timedelta(
            days=days_to_weekday + (n - 1) * 7
        )

    def last_weekday(year, month, weekday):
        last_day = (
            pd.Timestamp(year, month, 1)
            + pd.offsets.MonthEnd(1)
        )

        days_back = (
            last_day.weekday() - weekday
        ) % 7

        return last_day - pd.Timedelta(
            days=days_back
        )

    holiday_dates = set()

    for year in range(
        df["Datetime"].dt.year.min(),
        df["Datetime"].dt.year.max() + 1
    ):
        holiday_dates.add(
            pd.Timestamp(year, 1, 1)
        )

        holiday_dates.add(
            nth_weekday(year, 1, 0, 3)
        )

        holiday_dates.add(
            nth_weekday(year, 2, 0, 3)
        )

        holiday_dates.add(
            last_weekday(year, 5, 0)
        )

        holiday_dates.add(
            pd.Timestamp(year, 7, 4)
        )

        holiday_dates.add(
            nth_weekday(year, 9, 0, 1)
        )

        thanksgiving = nth_weekday(
            year, 11, 3, 4
        )

        holiday_dates.add(thanksgiving)

        holiday_dates.add(
            thanksgiving + pd.Timedelta(days=1)
        )

        holiday_dates.add(
            pd.Timestamp(year, 12, 24)
        )

        holiday_dates.add(
            pd.Timestamp(year, 12, 25)
        )

        holiday_dates.add(
            pd.Timestamp(year, 12, 31)
        )

    df["Is_Holiday"] = (
        df["Datetime"]
        .dt.normalize()
        .isin(holiday_dates)
        .astype(int)
    )

    df["Lag_1"] = df[target].shift(1)
    df["Lag_24"] = df[target].shift(24)
    df["Lag_168"] = df[target].shift(168)

    df["Rolling_Mean_24"] = (
        df[target]
        .shift(1)
        .rolling(24, min_periods=24)
        .mean()
    )

    df["Rolling_Std_24"] = (
        df[target]
        .shift(1)
        .rolling(24, min_periods=24)
        .std()
    )

    df["Rolling_Mean_168"] = (
        df[target]
        .shift(1)
        .rolling(168, min_periods=168)
        .mean()
    )

    df["Rolling_Std_168"] = (
        df[target]
        .shift(1)
        .rolling(168, min_periods=168)
        .std()
    )

    return df






# ------------------------------------------------------------
# Generate 30-day forecast
# ------------------------------------------------------------

st.subheader("30-Day Power Demand Forecast")

forecast_days = 30
forecast_hours = forecast_days * 24

if st.button("Generate Forecast"):

    history = data.copy()

    future_predictions = []

    for _ in range(forecast_hours):

        next_datetime = (
            history["Datetime"].iloc[-1]
            + pd.Timedelta(hours=1)
        )

        # Create temporary row
        new_row = pd.DataFrame({
            "Datetime": [next_datetime],
            target: [np.nan]
        })

        temp = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        # Generate features
        temp = create_features(temp)

        # Get latest feature row
        X_future = temp[features].iloc[[-1]]

        # Predict
        prediction = model.predict(X_future)[0]

        # Store prediction
        new_row[target] = prediction

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        future_predictions.append({
            "Datetime": next_datetime,
            "Forecast_MW": prediction
        })

    forecast_df = pd.DataFrame(future_predictions)

    st.success("30-day forecast generated successfully.")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )




    # --------------------------------------------------------
    # Forecast graph
    # --------------------------------------------------------

    st.subheader("Forecast Trend")

    st.line_chart(
        forecast_df.set_index("Datetime")["Forecast_MW"]
    )

    # --------------------------------------------------------
    # Download forecast
    # --------------------------------------------------------

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="30_day_power_demand_forecast.csv",
        mime="text/csv"
    )