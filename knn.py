import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

with open('features_30_sec.csv', 'r') as f:
  df = pd.read_csv(f)

# Extract features and labels
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameters to search
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Perform grid search with cross-validation
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train KNN classifier with best hyperparameters
knn = grid_search.best_estimator_
knn.fit(X_train, y_train)
# Predict labels for test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)

# Print best hyperparameters
print('Best hyperparameters:', grid_search.best_params_)