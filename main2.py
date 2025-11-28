
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Sample Customer Churn Model")

# Larger dummy dataset (more realistic)
data = pd.DataFrame({
    'usage': [10, 20, 30, 40, 50, 22, 33, 60],
    'complaints': [0, 1, 0, 1, 0, 1, 0, 1],
    'churn':      [0, 1, 0, 1, 0, 1, 0, 1]  # balanced 0 and 1
})

X = data[['usage', 'complaints']]
y = data['churn']

# Ensure balanced splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Predictions:", pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, pred))
