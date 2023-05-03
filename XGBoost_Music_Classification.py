import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
#from google.colab import drive

# Load the data from a CSV file
#drive.mount('/content/drive')
with open('features_30_sec.csv', 'r') as f:
    data = pd.read_csv(f)

# Split the data into features and labels
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create multiple XGBoost models with different hyperparameters
model1 = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100)
model2 = XGBClassifier(max_depth=10, learning_rate=0.05, n_estimators=200)
model3 = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=50)

# Combine the models into a VotingClassifier
ensemble = VotingClassifier(
    estimators=[('model1', model1), ('model2', model2), ('model3', model3)],
    voting='soft'
)

# Train the ensemble model
ensemble.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the accuracy of the model
accuracy = (cm.diagonal().sum() / cm.sum()) * 100

y_test_series = pd.Series(y_test)

# Calculate the total number of test samples from each class
class_totals = y_test_series.value_counts().sort_index().values


# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=max(class_totals), annot_kws={"fontsize":14})

# Add labels, title, and ticks
plt.title('Confusion Matrix for XGBoost (Accuracy: {:.2f}%)'.format(accuracy))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], rotation=90)
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], rotation=0)

plt.show()