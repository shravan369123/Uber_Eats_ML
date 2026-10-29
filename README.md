# Uber Eats Operational & Customer Intelligence System

An end-to-end Machine Learning and analytics platform designed to optimize food delivery operations, segment marketplace participants, predict fulfillment delays, and forecast demand spikes.

---

## Technical Stack & Architecture
- **Language & Runtime:** Python 3.10+
- **Data Engineering:** Pandas, NumPy
- **Unsupervised ML:** Scikit-Learn (K-Means, DBSCAN, PCA)
- **Natural Language Processing:** NLTK, VADER, Scikit-Learn (TF-IDF, Naive Bayes)
- **Supervised Regression:** Scikit-Learn (Random Forest, Ridge, Lasso)
- **Time-Series Forecasting:** Statsmodels (ADF, ARIMA), Facebook Prophet
- **Visualizations:** Matplotlib, Seaborn

---

## Project Folder Structure

```text
uber-eats-ml-system/
├── data/
│   ├── raw/                  # Raw synthetic CSV datasets (Customers, Orders, Deliveries, etc.)
│   └── processed/            # Feature-engineered analytical datasets & segmentations
├── models/                   # Saved ML models & binary artifacts
├── notebooks/                # Exploratory Jupyter Notebooks
├── reports/
│   ├── figures/              # Saved model evaluation charts & forecast plots
│   └── pipeline_executive_summary.txt
├── src/
│   ├── data/                 # Data generation scripts
│   ├── features/             # Feature extraction & scaling pipelines
│   ├── models/               # Model training, sentiment, and forecasting modules
│   └── pipeline.py           # Master end-to-end pipeline runner
├── data_dictionary.md        # Comprehensive schema documentation
├── requirements.txt          # Environment dependencies
└── README.md                 # Project technical documentation

## Key Findings & Business Impact

1. **Customer & Restaurant Segmentation:** Identified 3 primary behavioral clusters (Occasional Budget, Premium Frequent Diners, Late-Night Diners). Premium Diners generate over $35\%$ of platform GMV.
2. **Sentiment Analysis:** TF-IDF n-gram drivers revealed that delivery delays exceeding 35 minutes account for over 70% of 1-star reviews citing "cold food" and "soggy" packaging.
3. **Delivery Time Prediction:** Random Forest Regressor outperformed linear models ($R^2 = 0.915$, MAE = 1.82 mins). Distance, traffic density, and rain/snow weather act as top delay predictors.
4. **Demand Forecasting:** Facebook Prophet achieved a 9.1% MAPE on hourly order volumes. Captured recurring weekend dinner spikes (18:00–21:00) to optimize driver dispatching.

