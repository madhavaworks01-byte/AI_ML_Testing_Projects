# robust version
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Sample Customer Churn Model")
data = pd.DataFrame({
    'usage': [10, 20, 30, 40],
    'complaints': [0, 1, 0, 1],
    'churn': [0, 1, 0, 1]   # ensure both classes exist
})

X = data[['usage', 'complaints']]
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

unique_classes = np.unique(y_train)
if unique_classes.shape[0] < 2:
    print("Training set contains only one class:", unique_classes)
    print("You must provide at least 2 classes to train the model. Try different split or add more data.")
else:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
