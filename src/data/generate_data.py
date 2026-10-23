import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

N_CUSTOMERS = 1000
N_RESTAURANTS = 150
N_DRIVERS = 200
N_ORDERS = 10000

# 1. Customers Dataset
customer_ids = [f"CUST_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
loyalty_tiers = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=N_CUSTOMERS, p=[0.5, 0.3, 0.15, 0.05])
signup_channels = np.random.choice(['Organic', 'Referral', 'Paid Ads', 'Social Media'], size=N_CUSTOMERS, p=[0.4, 0.25, 0.2, 0.15])
signup_dates = [datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 700))) for _ in range(N_CUSTOMERS)]

df_customers = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'loyalty_tier': loyalty_tiers,
    'signup_channel': signup_channels
})

# 2. Restaurants Dataset
restaurant_ids = [f"REST_{i:04d}" for i in range(1, N_RESTAURANTS + 1)]
cuisines = ['American', 'Italian', 'Indian', 'Japanese', 'Mexican', 'Chinese', 'Thai', 'Healthy']
rest_cuisines = np.random.choice(cuisines, size=N_RESTAURANTS)
rest_ratings = np.round(np.random.uniform(3.0, 5.0, size=N_RESTAURANTS), 1)
commissions = np.random.choice([0.15, 0.20, 0.25, 0.30], size=N_RESTAURANTS, p=[0.1, 0.4, 0.4, 0.1])

df_restaurants = pd.DataFrame({
    'restaurant_id': restaurant_ids,
    'cuisine_type': rest_cuisines,
    'restaurant_rating': rest_ratings,
    'commission_rate': commissions,
    'is_active': True
})

# 3. Drivers Dataset
driver_ids = [f"DRV_{i:04d}" for i in range(1, N_DRIVERS + 1)]
vehicles = np.random.choice(['Bicycle', 'Scooter', 'Motorcycle', 'Car'], size=N_DRIVERS, p=[0.1, 0.3, 0.4, 0.2])
driver_ratings = np.round(np.random.uniform(4.0, 5.0, size=N_DRIVERS), 1)

df_drivers = pd.DataFrame({
    'driver_id': driver_ids,
    'vehicle_type': vehicles,
    'driver_rating': driver_ratings,
    'total_deliveries_completed': np.random.randint(10, 2000, size=N_DRIVERS)
})

# 4. Orders Dataset
order_ids = [f"ORD_{i:07d}" for i in range(1, N_ORDERS + 1)]
cust_sample = np.random.choice(customer_ids, size=N_ORDERS)
rest_sample = np.random.choice(restaurant_ids, size=N_ORDERS)

start_date = datetime(2026, 1, 1, 0, 0, 0)
timestamps = []
hour_probs = np.array([0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.03, 0.03, 0.04, 0.08, 0.12, 0.06, 0.03, 0.03, 0.04, 0.08, 0.15, 0.12, 0.06, 0.03, 0.02, 0.01])
hour_probs /= hour_probs.sum()

for _ in range(N_ORDERS):
    day_offset = np.random.randint(0, 290)
    hour = np.random.choice(range(24), p=hour_probs)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    ts = start_date + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute), seconds=int(second))
    timestamps.append(ts)

subtotals = np.round(np.random.exponential(scale=25.0) + 10.0, 2)
order_statuses = np.random.choice(['Completed', 'Cancelled'], size=N_ORDERS, p=[0.95, 0.05])

df_orders = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': cust_sample,
    'restaurant_id': rest_sample,
    'order_timestamp': timestamps,
    'subtotal_amount': subtotals,
    'order_status': order_statuses
})

df_orders['is_weekend'] = df_orders['order_timestamp'].dt.dayofweek.isin([5, 6])
df_orders['is_late_night'] = df_orders['order_timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4])

completed_orders = df_orders[df_orders['order_status'] == 'Completed']['order_id'].tolist()

# 5. Deliveries Dataset
delivery_records = []
weathers = ['Clear', 'Cloudy', 'Rain', 'Snow']
traffics = ['Low', 'Medium', 'High', 'Jam']

for ord_id in completed_orders:
    prep_time = max(5.0, round(np.random.normal(loc=18, scale=5), 1))
    distance = round(np.random.exponential(scale=3.5) + 0.5, 2)
    weather = np.random.choice(weathers, p=[0.6, 0.25, 0.1, 0.05])
    traffic = np.random.choice(traffics, p=[0.4, 0.35, 0.18, 0.07])
    
    base_time = prep_time + (distance * 4.0)
    weather_mult = 1.3 if weather in ['Rain', 'Snow'] else 1.0
    traffic_mult = 1.4 if traffic in ['High', 'Jam'] else 1.0
    
    delivery_time = max(10.0, round(base_time * weather_mult * traffic_mult + np.random.normal(0, 3), 1))
    
    delivery_records.append({
        'delivery_id': f"DEL_{ord_id.split('_')[1]}",
        'order_id': ord_id,
        'driver_id': np.random.choice(driver_ids),
        'prep_time_min': prep_time,
        'delivery_distance_km': distance,
        'weather': weather,
        'traffic_density': traffic,
        'delivery_time_min': delivery_time
    })

df_deliveries = pd.DataFrame(delivery_records)

# 6. Payments Dataset
payment_records = []
methods = ['Credit Card', 'Uber Cash', 'Apple Pay', 'PayPal']

for ord_id in completed_orders:
    subtotal = df_orders.loc[df_orders['order_id'] == ord_id, 'subtotal_amount'].values[0]
    method = np.random.choice(methods, p=[0.5, 0.2, 0.2, 0.1])
    tip_pct = np.random.choice([0.0, 0.10, 0.15, 0.18, 0.20], p=[0.2, 0.2, 0.3, 0.2, 0.1])
    tip = round(subtotal * tip_pct, 2)
    
    payment_records.append({
        'payment_id': f"PAY_{ord_id.split('_')[1]}",
        'order_id': ord_id,
        'payment_method': method,
        'tip_amount': tip,
        'total_paid': round(subtotal + tip + 3.99, 2)
    })

df_payments = pd.DataFrame(payment_records)

# 7. Reviews Dataset
pos_reviews = [
    "Food arrived super fast, piping hot and absolutely delicious!",
    "Great driver, polite delivery, food fresh and well packed.",
    "Amazing meal! The order came quicker than expected. 5 stars!",
    "Driver followed instructions perfectly. Great Uber Eats service.",
    "Best experience so far! Fresh sushi and intact packaging."
]
neg_reviews = [
    "Cold food, delayed by 40 minutes! Terrible experience.",
    "Order was completely messed up, missing items, and late delivery.",
    "Food was spilled inside the bag. Driver dropped it and left.",
    "Rude driver and lukewarm food. Unacceptable quality for the price.",
    "Extremely slow delivery. The food was soggy and cold."
]
neu_reviews = [
    "Average delivery time. Food was okay, nothing special.",
    "Order was correct. Average portion sizes and standard packaging.",
    "Food arrived on time but could have been warmer."
]

review_records = []
sampled_completed = np.random.choice(completed_orders, size=int(len(completed_orders) * 0.4), replace=False)

for ord_id in sampled_completed:
    cust_id = df_orders.loc[df_orders['order_id'] == ord_id, 'customer_id'].values[0]
    rest_id = df_orders.loc[df_orders['order_id'] == ord_id, 'restaurant_id'].values[0]
    score = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.15, 0.25, 0.4])
    
    text = random.choice(neg_reviews) if score in [1, 2] else (random.choice(neu_reviews) if score == 3 else random.choice(pos_reviews))
        
    review_records.append({
        'review_id': f"REV_{ord_id.split('_')[1]}",
        'order_id': ord_id,
        'customer_id': cust_id,
        'restaurant_id': rest_id,
        'rating_score': score,
        'review_text': text
    })

df_reviews = pd.DataFrame(review_records)

df_customers.to_csv(f"{DATA_DIR}/customers.csv", index=False)
df_restaurants.to_csv(f"{DATA_DIR}/restaurants.csv", index=False)
df_drivers.to_csv(f"{DATA_DIR}/drivers.csv", index=False)
df_orders.to_csv(f"{DATA_DIR}/orders.csv", index=False)
df_deliveries.to_csv(f"{DATA_DIR}/deliveries.csv", index=False)
df_payments.to_csv(f"{DATA_DIR}/payments.csv", index=False)
df_reviews.to_csv(f"{DATA_DIR}/reviews.csv", index=False)

print("SUCCESS: All 7 synthetic datasets generated in data/raw/")