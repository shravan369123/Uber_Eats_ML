import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

df_orders = pd.read_csv("data/raw/orders.csv")
df_reviews = pd.read_csv("data/raw/reviews.csv")
df_deliveries = pd.read_csv("data/raw/deliveries.csv")
df_restaurants = pd.read_csv("data/raw/restaurants.csv")
df_customers = pd.read_csv("data/raw/customers.csv")

# -------------------------------------------------------------
# 1. Feature Engineering: Customers
# -------------------------------------------------------------
cust_orders = df_orders.groupby('customer_id').agg(
    total_orders=('order_id', 'count'),
    total_spending=('subtotal_amount', 'sum'),
    avg_order_value=('subtotal_amount', 'mean'),
    weekend_orders=('is_weekend', 'sum'),
    late_night_orders=('is_late_night', 'sum')
).reset_index()

cust_reviews = df_reviews.groupby('customer_id').agg(
    avg_rating_given=('rating_score', 'mean')
).reset_index()

cust_features = pd.merge(cust_orders, cust_reviews, on='customer_id', how='left')
cust_features['avg_rating_given'] = cust_features['avg_rating_given'].fillna(cust_features['avg_rating_given'].median())

# Ordering frequency: total orders divided by span of platform activity (normalized)
cust_features['ordering_frequency'] = np.round(cust_features['total_orders'] / 9.0, 2)

cust_features.to_csv("data/processed/customer_features.csv", index=False)

# -------------------------------------------------------------
# 2. Feature Engineering: Restaurants
# -------------------------------------------------------------
rest_orders = df_orders.groupby('restaurant_id').agg(
    total_orders=('order_id', 'count'),
    revenue=('subtotal_amount', 'sum'),
    cancelled_orders=('order_status', lambda x: (x == 'Cancelled').sum())
).reset_index()

rest_orders['cancellation_rate'] = np.round(rest_orders['cancelled_orders'] / rest_orders['total_orders'], 4)

rest_deliv = df_orders.merge(df_deliveries, on='order_id', how='inner')
rest_deliv_agg = rest_deliv.groupby('restaurant_id').agg(
    avg_prep_time=('prep_time_min', 'mean'),
    avg_delivery_time=('delivery_time_min', 'mean')
).reset_index()

rest_features = pd.merge(rest_orders, rest_deliv_agg, on='restaurant_id', how='left')
rest_features = pd.merge(rest_features, df_restaurants[['restaurant_id', 'restaurant_rating']], on='restaurant_id', how='left')
rest_features['avg_prep_time'] = rest_features['avg_prep_time'].fillna(rest_features['avg_prep_time'].median())
rest_features['avg_delivery_time'] = rest_features['avg_delivery_time'].fillna(rest_features['avg_delivery_time'].median())
rest_features.drop(columns=['cancelled_orders'], inplace=True)

rest_features.to_csv("data/processed/restaurant_features.csv", index=False)

# -------------------------------------------------------------
# 3. Model Training & Clustering: Customer Segmentation
# -------------------------------------------------------------
cust_clustering_cols = [
    'total_orders', 'avg_order_value', 'total_spending',
    'ordering_frequency', 'avg_rating_given', 'weekend_orders', 'late_night_orders'
]

X_cust = cust_features[cust_clustering_cols]
scaler_cust = StandardScaler()
X_cust_scaled = scaler_cust.fit_transform(X_cust)

# Elbow & Silhouette Evaluation for K-Means (k=2 to 8)
inertia = []
sil_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cust_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cust_scaled, km.labels_))

# Save Elbow & Silhouette Plot
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(k_range, inertia, 'bo-', label='Inertia (Elbow)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='b')

ax2 = ax1.twinx()
ax2.plot(k_range, sil_scores, 'r*--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
plt.title('Customer Segmentation: Elbow & Silhouette Plot')
plt.tight_layout()
plt.savefig('reports/figures/customer_elbow_silhouette.png')
plt.close()

# Fit Optimal K-Means (k=3)
optimal_k = 3
kmeans_cust = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cust_features['kmeans_cluster'] = kmeans_cust.fit_predict(X_cust_scaled)

# DBSCAN
dbscan_cust = DBSCAN(eps=1.5, min_samples=5)
cust_features['dbscan_cluster'] = dbscan_cust.fit_predict(X_cust_scaled)

# PCA for 2D Visualization
pca = PCA(n_components=2)
cust_pca = pca.fit_transform(X_cust_scaled)
cust_features['pca1'] = cust_pca[:, 0]
cust_features['pca2'] = cust_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=cust_features, 
    x='pca1', 
    y='pca2', 
    hue='kmeans_cluster', 
    palette='Set1', 
    alpha=0.8
)
plt.title(f'Customer Clusters Visualized with PCA (Explained Variance: {np.sum(pca.explained_variance_ratio_):.2f})')
plt.tight_layout()
plt.savefig('reports/figures/customer_clusters_pca.png')
plt.close()

# Save final segmented dataset
cust_features.to_csv("data/processed/customer_features_segmented.csv", index=False)

print("SUCCESS: Day 2 Feature Engineering and Clustering Completed.")
print(f"Customer Silhouette Score for k={optimal_k}: {silhouette_score(X_cust_scaled, kmeans_cust.labels_):.4f}")
print("DBSCAN Noise Points Identified:", (cust_features['dbscan_cluster'] == -1).sum())