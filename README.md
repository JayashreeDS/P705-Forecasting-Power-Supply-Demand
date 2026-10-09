# P705 - Forecasting Power Supply Demand

## Project Overview
This project focuses on forecasting hourly power demand using historical electricity consumption data. It analyzes demand patterns and uses machine learning to predict future power demand.

## Objectives
- Analyze historical hourly power demand.
- Identify daily, weekly, and seasonal trends.
- Build and evaluate forecasting models.
- Develop an interactive dashboard to visualize forecasts.

## Dataset
The project uses the PJM hourly power demand dataset.

- Target variable: Power demand (MW)
- Data frequency: Hourly

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook

## Models
- SARIMA
- Random Forest
- XGBoost

## Final Model Performance
The final XGBoost model achieved the following backtesting results:

| Metric | Result |
|---|---:|
| MAE | 61.00 MW |
| RMSE | 77.51 MW |
| MAPE | 1.00% |

These results are based on the project's evaluation setup; performance may vary on future data.

## Streamlit Dashboard
The dashboard provides:
- Historical power demand visualization.
- Forecasts for 7, 15, and 30 days.
- Forecast summaries and downloadable results.

## Project Structure
- `dataset/` - Original dataset
- `notebooks/` - Data analysis and model development
- `reports/` - Analysis results and evaluation reports
- `deployment/` - Streamlit application and saved model files

## How to Run the Dashboard

1. Clone the repository:

   ```bash
   git clone https://github.com/JayashreeDS/P705-Forecasting-Power-Supply-Demand.git
