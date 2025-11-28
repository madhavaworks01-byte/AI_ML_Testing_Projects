# save this as house_price_prediction/main2.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

print("House Price Prediction - Simple Linear Regression")

# Improved dummy dataset (area in sq.ft, rooms, age_of_house(years), price in 1000s)
data = pd.DataFrame({
    'area': [850, 900, 1200, 1500, 1800, 2000, 2300, 2600, 3000, 3200],
    'rooms': [2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
    'age':   [20, 18, 15, 10, 8, 6, 5, 3, 2, 1],
    'price': [35, 38, 55, 70, 85, 95, 110, 130, 150, 170]  # prices in 1000s
})

# Features and target
X = data[['area', 'rooms', 'age']]
y = data['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print("\nTest set predictions vs actuals:")
for p, a in zip(pred.round(2), y_test.values):
    print("Predicted:", p, "Actual:", a)

print(f"\nR2 score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f} (in 1000s)")
