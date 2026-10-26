
import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Create sample house dataset
# --------------------------------------------------

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
    "price": [
        2500000, 3200000, 3500000, 4800000, 5200000,
        6000000, 6500000, 7500000, 2300000, 3800000,
        4500000, 5000000, 6800000, 7200000, 8500000,
        3000000, 4000000, 4700000, 7000000, 8000000
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# 2. Prepare features and target
# --------------------------------------------------

X = df[["area", "bedrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# 3. Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House Price Prediction")

# --------------------------------------------------
# 4. Define three machine learning models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

results = []
best_model = None
best_model_name = None
best_mae = float("inf")

# --------------------------------------------------
# 5. Train, evaluate and track each model
# --------------------------------------------------

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on unseen test data
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log parameters and metrics in MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", "area,bedrooms,age")

        for param_name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(param_name, param_value)

        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("MSE", float(mse))
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("R2_Score", float(r2))

        # Save the model as an MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        # Display results
        print(f"\nModel: {model_name}")
        print(f"MAE: ₹{mae:,.2f}")
        print(f"RMSE: ₹{rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        # Select the model with the lowest test MAE
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = model_name

# --------------------------------------------------
# 6. Display model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv("model_comparison.csv", index=False)

# --------------------------------------------------
# 7. Save the best model for the Streamlit app
# --------------------------------------------------

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\n" + "=" * 55)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Best model: {best_model_name}")
print(f"Best test MAE: ₹{best_mae:,.2f}")
print("Saved model: house_price_model.pkl")
print("Saved comparison: model_comparison.csv")
print("MLflow tracking directory: ./mlruns")
print("=" * 55)
