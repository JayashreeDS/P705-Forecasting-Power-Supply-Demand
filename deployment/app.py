import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PJM Power Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("xgboost_model.pkl")
    features = joblib.load("feature_info.pkl")

    return model, features


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "../reports/PJMW_EDA_dataset.csv"
    )

    data["Datetime"] = pd.to_datetime(
        data["Datetime"]
    )

    data = data.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    return data


model, features = load_model()
data = load_data()


# =========================================================
# SELECT TARGET COLUMN
# =========================================================

if "PJMW_MW_Clean" in data.columns:

    target = "PJMW_MW_Clean"

elif "PJMW_MW" in data.columns:

    target = "PJMW_MW"

else:

    st.error(
        "Power demand column not found in the dataset."
    )

    st.stop()


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # Calendar features
    # -----------------------------------------------------

    df["Hour"] = df["Datetime"].dt.hour

    df["DayOfWeek"] = (
        df["Datetime"].dt.dayofweek
    )

    df["Month"] = (
        df["Datetime"].dt.month
    )

    df["DayOfYear"] = (
        df["Datetime"].dt.dayofyear
    )

    df["WeekOfYear"] = (
        df["Datetime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["Quarter"] = (
        df["Datetime"].dt.quarter
    )

    df["Is_Weekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)


    # -----------------------------------------------------
    # Holiday functions
    # -----------------------------------------------------

    def nth_weekday(
        year,
        month,
        weekday,
        n
    ):

        first_day = pd.Timestamp(
            year,
            month,
            1
        )

        days_to_weekday = (
            weekday - first_day.weekday()
        ) % 7

        return (
            first_day
            + pd.Timedelta(
                days=days_to_weekday
                + (n - 1) * 7
            )
        )


    def last_weekday(
        year,
        month,
        weekday
    ):

        last_day = (
            pd.Timestamp(
                year,
                month,
                1
            )
            + pd.offsets.MonthEnd(1)
        )

        days_back = (
            last_day.weekday()
            - weekday
        ) % 7

        return (
            last_day
            - pd.Timedelta(
                days=days_back
            )
        )


    # -----------------------------------------------------
    # Holiday dates
    # -----------------------------------------------------

    holiday_dates = set()

    min_year = (
        df["Datetime"]
        .dt.year
        .min()
    )

    max_year = (
        df["Datetime"]
        .dt.year
        .max()
    )

    for year in range(
        min_year,
        max_year + 2
    ):

        # New Year's Day
        holiday_dates.add(
            pd.Timestamp(
                year,
                1,
                1
            )
        )

        # Martin Luther King Jr. Day
        holiday_dates.add(
            nth_weekday(
                year,
                1,
                0,
                3
            )
        )

        # Presidents' Day
        holiday_dates.add(
            nth_weekday(
                year,
                2,
                0,
                3
            )
        )

        # Memorial Day
        holiday_dates.add(
            last_weekday(
                year,
                5,
                0
            )
        )

        # Independence Day
        holiday_dates.add(
            pd.Timestamp(
                year,
                7,
                4
            )
        )

        # Labor Day
        holiday_dates.add(
            nth_weekday(
                year,
                9,
                0,
                1
            )
        )

        # Thanksgiving
        thanksgiving = nth_weekday(
            year,
            11,
            3,
            4
        )

        holiday_dates.add(
            thanksgiving
        )

        # Day after Thanksgiving
        holiday_dates.add(
            thanksgiving
            + pd.Timedelta(days=1)
        )

        # Christmas Eve
        holiday_dates.add(
            pd.Timestamp(
                year,
                12,
                24
            )
        )

        # Christmas
        holiday_dates.add(
            pd.Timestamp(
                year,
                12,
                25
            )
        )

        # New Year's Eve
        holiday_dates.add(
            pd.Timestamp(
                year,
                12,
                31
            )
        )


    df["Is_Holiday"] = (
        df["Datetime"]
        .dt.normalize()
        .isin(holiday_dates)
        .astype(int)
    )


    # -----------------------------------------------------
    # Lag features
    # -----------------------------------------------------

    df["Lag_1"] = (
        df[target]
        .shift(1)
    )

    df["Lag_24"] = (
        df[target]
        .shift(24)
    )

    df["Lag_168"] = (
        df[target]
        .shift(168)
    )


    # -----------------------------------------------------
    # Rolling features
    # -----------------------------------------------------

    df["Rolling_Mean_24"] = (
        df[target]
        .shift(1)
        .rolling(
            24,
            min_periods=24
        )
        .mean()
    )

    df["Rolling_Std_24"] = (
        df[target]
        .shift(1)
        .rolling(
            24,
            min_periods=24
        )
        .std()
    )

    df["Rolling_Mean_168"] = (
        df[target]
        .shift(1)
        .rolling(
            168,
            min_periods=168
        )
        .mean()
    )

    df["Rolling_Std_168"] = (
        df[target]
        .shift(1)
        .rolling(
            168,
            min_periods=168
        )
        .std()
    )

    return df


# =========================================================
# TITLE
# =========================================================

st.title(
    "⚡ PJM Power Demand Forecasting"
)

st.write(
    "Forecasting hourly power demand "
    "using an XGBoost machine learning model."
)

st.success(
    "Model and dataset loaded successfully."
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

st.subheader(
    "📌 Project Overview"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "XGBoost"
    )

with col2:

    st.metric(
        "Forecast Period",
        "Up to 30 Days"
    )

with col3:

    st.metric(
        "Frequency",
        "Hourly"
    )


# =========================================================
# HISTORICAL POWER DEMAND
# =========================================================

st.subheader(
    "📊 Historical Power Demand"
)

st.line_chart(
    data.set_index("Datetime")[target]
)

st.write(
    f"Historical data available from "
    f"{data['Datetime'].min().date()} "
    f"to "
    f"{data['Datetime'].max().date()}."
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader(
    "📈 Model Performance"
)

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

with metric_col1:

    st.metric(
        "MAE",
        "61.00 MW"
    )

with metric_col2:

    st.metric(
        "RMSE",
        "77.51 MW"
    )

with metric_col3:

    st.metric(
        "MAPE",
        "1.00%"
    )

st.caption(
    "Performance based on the 30-day backtest."
)


# =========================================================
# FORECAST SETTINGS
# =========================================================

st.subheader(
    "🔮 Generate Power Demand Forecast"
)

forecast_days = st.selectbox(
    "Select forecast period",
    [7, 15, 30],
    index=2
)

st.write(
    f"The model will generate an hourly forecast "
    f"for the next {forecast_days} days."
)


# =========================================================
# GENERATE FORECAST
# =========================================================

if st.button(
    "🚀 Generate Forecast",
    type="primary"
):

    with st.spinner(
        "Generating power demand forecast..."
    ):

        history = data.copy()

        future_predictions = []

        forecast_hours = (
            forecast_days * 24
        )

        for _ in range(forecast_hours):

            # Find next hour
            next_datetime = (
                history["Datetime"].iloc[-1]
                + pd.Timedelta(hours=1)
            )

            # Create new row
            new_row = pd.DataFrame(
                {
                    "Datetime": [
                        next_datetime
                    ],
                    target: [
                        np.nan
                    ]
                }
            )

            # Add new row temporarily
            temp = pd.concat(
                [
                    history,
                    new_row
                ],
                ignore_index=True
            )

            # Create required features
            temp = create_features(
                temp
            )

            # Select model features
            X_future = (
                temp[features]
                .iloc[[-1]]
            )

            # Generate prediction
            prediction = model.predict(
                X_future
            )[0]

            # Add prediction to history
            new_row[target] = (
                prediction
            )

            history = pd.concat(
                [
                    history,
                    new_row
                ],
                ignore_index=True
            )

            # Store prediction
            future_predictions.append(
                {
                    "Datetime":
                        next_datetime,

                    "Forecast_MW":
                        prediction
                }
            )


        # Convert predictions to DataFrame
        forecast_df = pd.DataFrame(
            future_predictions
        )


    st.success(
        f"{forecast_days}-day forecast "
        "generated successfully!"
    )


    # =====================================================
    # FORECAST SUMMARY
    # =====================================================

    st.subheader(
        "📌 Forecast Summary"
    )

    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )

    with summary_col1:

        st.metric(
            "Average Demand",
            f"{forecast_df['Forecast_MW'].mean():,.0f} MW"
        )

    with summary_col2:

        st.metric(
            "Peak Demand",
            f"{forecast_df['Forecast_MW'].max():,.0f} MW"
        )

    with summary_col3:

        st.metric(
            "Minimum Demand",
            f"{forecast_df['Forecast_MW'].min():,.0f} MW"
        )


    # =====================================================
    # FORECAST TREND
    # =====================================================

    st.subheader(
        "📉 Forecast Trend"
    )

    st.line_chart(
        forecast_df.set_index(
            "Datetime"
        )["Forecast_MW"]
    )


    # =====================================================
    # FORECAST TABLE
    # =====================================================

    st.subheader(
        "📋 Forecasted Power Demand"
    )

    display_df = forecast_df.copy()

    display_df["Datetime"] = (
        display_df["Datetime"]
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    display_df["Forecast_MW"] = (
        display_df["Forecast_MW"]
        .round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DOWNLOAD FORECAST
    # =====================================================

    csv = forecast_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Forecast CSV",

        data=csv,

        file_name=(
            f"{forecast_days}_day_"
            "power_demand_forecast.csv"
        ),

        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PJM Power Demand Forecasting | "
    "Machine Learning Project | "
    "XGBoost"
)