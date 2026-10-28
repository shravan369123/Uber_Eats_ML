import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Construction & Feature Merging
# -------------------------------------------------------------
df_orders = pd.read_csv("data/raw/orders.csv")
df_deliveries = pd.read_csv("data/raw/deliveries.csv")
df_payments = pd.read_csv("data/raw/payments.csv")
df_restaurants = pd.read_csv("data/raw/restaurants.csv")
df_cust_seg = pd.read_csv("data/processed/customer_features_segmented.csv")

# Merge into analytical regression dataframe
df_reg = df_orders.merge(df_deliveries, on='order_id', how='inner')
df_reg = df_reg.merge(df_payments[['order_id', 'tip_amount']], on='order_id', how='inner')
df_reg = df_reg.merge(df_restaurants[['restaurant_id', 'restaurant_rating', 'cuisine_type']], on='restaurant_id', how='left')
df_reg = df_reg.merge(df_cust_seg[['customer_id', 'kmeans_cluster']], on='customer_id', how='left')

# Save regression-ready analytical dataset
df_reg.to_csv("data/processed/regression_analytics.csv", index=False)

# -------------------------------------------------------------
# 2. Setup Features & Preprocessing Pipeline
# -------------------------------------------------------------
num_features = ['subtotal_amount', 'prep_time_min', 'delivery_distance_km', 'restaurant_rating']
cat_features = ['weather', 'traffic_density', 'cuisine_type', 'is_weekend', 'is_late_night']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)

# Target A: Delivery Time
X = df_reg[num_features + cat_features]
y = df_reg['delivery_time_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess matrices
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# -------------------------------------------------------------
# 3. Model Training & Evaluation
# -------------------------------------------------------------
models = {
    'Baseline Linear Regression': LinearRegression(),
    'Ridge Regression (L2)': Ridge(alpha=1.0),
    'Lasso Regression (L1)': Lasso(alpha=0.1),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_proc, y_train)
    preds = model.predict(X_test_proc)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({
        'Model': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2 Score': round(r2, 4)
    })

df_results = pd.DataFrame(results)
print("=== Delivery Time Model Comparison ===")
print(df_results.to_string(index=False))

# -------------------------------------------------------------
# 4. Feature Importance & Visualization (Random Forest)
# -------------------------------------------------------------
rf_model = models['Random Forest Regressor']

# Get transformed feature names
cat_encoder = preprocessor.named_transformers_['cat']
cat_encoded_names = cat_encoder.get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + cat_encoded_names

importances = rf_model.feature_importances_
df_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances (Delivery Time Prediction)')
plt.tight_layout()
plt.savefig('reports/figures/delivery_time_feature_importance.png')
plt.close()

# Actual vs Predicted Plot
rf_preds = rf_model.predict(X_test_proc)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_preds, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Delivery Time (min)')
plt.ylabel('Predicted Delivery Time (min)')
plt.title('Actual vs Predicted Delivery Time (Random Forest)')
plt.tight_layout()
plt.savefig('reports/figures/delivery_time_actual_vs_pred.png')
plt.close()

print("SUCCESS: Day 4 Regression Models Trained and Evaluated.")