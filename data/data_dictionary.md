# Uber Eats Analytics - Data Dictionary

This document details the schema and column definitions for the synthetic Uber Eats operational dataset.

---

## 1. Customers (`customers.csv`)
Contains customer demographic and account profile data.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `customer_id` | String | Unique identifier for each customer (Primary Key) |
| `signup_date` | Date | Date when the customer created their account |
| `loyalty_tier` | String | Customer loyalty classification (`Bronze`, `Silver`, `Gold`, `Platinum`) |
| `signup_channel` | String | Marketing acquisition channel (`Organic`, `Referral`, `Paid Ads`, `Social Media`) |

---

## 2. Restaurants (`restaurants.csv`)
Contains details on partner restaurants listed on the platform.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `restaurant_id` | String | Unique identifier for each restaurant (Primary Key) |
| `cuisine_type` | String | Primary cuisine served (`American`, `Italian`, `Indian`, `Japanese`, etc.) |
| `restaurant_rating` | Float | Average rating score of the restaurant (1.0 to 5.0) |
| `commission_rate` | Float | Platform commission fee percentage (e.g., 0.15, 0.20, 0.30) |
| `is_active` | Boolean | Account status flag (`True` = active, `False` = inactive) |

---

## 3. Drivers (`drivers.csv`)
Contains delivery partner profile information.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `driver_id` | String | Unique identifier for each delivery driver (Primary Key) |
| `vehicle_type` | String | Vehicle mode (`Bicycle`, `Scooter`, `Motorcycle`, `Car`) |
| `driver_rating` | Float | Driver performance rating (1.0 to 5.0) |
| `total_deliveries_completed` | Integer | Historical cumulative deliveries completed by driver |

---

## 4. Orders (`orders.csv`)
Contains order-level transaction details.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `order_id` | String | Unique identifier for each order transaction (Primary Key) |
| `customer_id` | String | Foreign key linking to `customers.csv` |
| `restaurant_id` | String | Foreign key linking to `restaurants.csv` |
| `order_timestamp` | Datetime | Exact timestamp when order was placed (`YYYY-MM-DD HH:MM:SS`) |
| `subtotal_amount` | Float | Order value excluding fees and taxes ($) |
| `order_status` | String | Status of order (`Completed`, `Cancelled`) |
| `is_weekend` | Boolean | True if order was placed on Saturday or Sunday |
| `is_late_night` | Boolean | True if order was placed between 23:00 and 04:59 |

---

## 5. Deliveries (`deliveries.csv`)
Contains fulfillment and logistical tracking metrics for completed orders.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `delivery_id` | String | Unique identifier for delivery fulfillment (Primary Key) |
| `order_id` | String | Foreign key linking to `orders.csv` |
| `driver_id` | String | Foreign key linking to `drivers.csv` |
| `prep_time_min` | Float | Restaurant food preparation time in minutes |
| `delivery_distance_km` | Float | Distance from restaurant to customer destination in kilometers |
| `weather` | String | Ambient weather condition (`Clear`, `Cloudy`, `Rain`, `Snow`) |
| `traffic_density` | String | Congestion level (`Low`, `Medium`, `High`, `Jam`) |
| `delivery_time_min` | Float | Total delivery fulfillment duration in minutes (Target Variable) |

---

## 6. Payments (`payments.csv`)
Contains transaction payment and driver tipping records.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `payment_id` | String | Unique payment transaction ID (Primary Key) |
| `order_id` | String | Foreign key linking to `orders.csv` |
| `payment_method` | String | Payment gateway used (`Credit Card`, `Uber Cash`, `Apple Pay`, `PayPal`) |
| `tip_amount` | Float | Tip paid to driver ($) |
| `total_paid` | Float | Total order payment charged including subtotal, fee, and tip ($) |

---

## 7. Reviews (`reviews.csv`)
Contains customer feedback and rating records used for NLP analysis.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `review_id` | String | Unique review identifier (Primary Key) |
| `order_id` | String | Foreign key linking to `orders.csv` |
| `customer_id` | String | Foreign key linking to `customers.csv` |
| `restaurant_id` | String | Foreign key linking to `restaurants.csv` |
| `rating_score` | Integer | Rating score given by customer (1 to 5 stars) |
| `review_text` | String | Free-form review feedback written by customer |