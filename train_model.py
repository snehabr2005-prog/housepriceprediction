
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create sample house data
data = {
    "area": [
        800, 1000, 1200, 1500, 1800,
        2000, 2200, 2500, 900, 1300,
        1600, 1900, 2100, 2400, 2800,
        1100, 1400, 1700, 2300, 2600
    ],
    "bedrooms": [
        1, 2, 2, 3, 3,
        3, 4, 4, 1, 2,
        3, 3, 4, 4, 5,
        2, 2, 3, 4, 5
    ],
    "age": [
        10, 5, 8, 3, 6,
        2, 4, 1, 15, 7,
        5, 9, 3, 2, 1,
        12, 6, 4, 3, 2
    ],
    # Illustrative prices in Indian rupees
    "price": [
        2500000, 3200000, 3500000, 4800000, 5200000,
        6000000, 6500000, 7500000, 2300000, 3800000,
        4500000, 5000000, 6800000, 7200000, 8500000,
        3000000, 4000000, 4700000, 7000000, 8000000
    ]
}

df = pd.DataFrame(data)

# 2. Select features and target
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Mean Absolute Error: ₹{mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# 6. Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_price_model.pkl")
