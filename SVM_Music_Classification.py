import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np
#from google.colab import drive
import pandas as pd

# Load the GTZAN dataset

#drive.mount('/content/drive')
with open('features_30_sec.csv', 'r') as f:
    data = pd.read_csv(f)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True, stratify=y)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM model and train it on the training set
svm_model = SVC(kernel='poly')
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the model
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
plt.title('Confusion Matrix for SVM (Accuracy: {:.2f}%)'.format(accuracy))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], rotation=90)
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], rotation=0)

plt.show()