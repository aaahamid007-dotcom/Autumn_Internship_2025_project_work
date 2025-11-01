# Q9 - Parkinson’s Dataset with Logistic Regression & Random Forest

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Parkinson's dataset
parkinsons_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df_parkinsons = pd.read_csv(parkinsons_url)

# Display the first few rows and summary statistics
display(df_parkinsons.head())
display(df_parkinsons.describe())

# Separate features and target
X_parkinsons = df_parkinsons.drop(['name', 'status'], axis=1)
y_parkinsons = df_parkinsons['status']

# Visualize the data (pairplot for a subset of features due to high dimensionality)
# Selecting a few features for visualization
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:RAP', 'status']
sns.pairplot(df_parkinsons[selected_features], hue="status")
plt.suptitle("Pairplot of Selected Parkinson's Features by Status", y=1.02)
plt.show()

# Visualize correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X_parkinsons.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap - Parkinson's Dataset")
plt.show()

# Train/Test Split for Parkinson's dataset
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_parkinsons, y_parkinsons, test_size=0.3, random_state=42, stratify=y_parkinsons)
print("\nParkinson's Dataset - Training samples:", X_train_p.shape[0])
print("Parkinson's Dataset - Test samples:", X_test_p.shape[0])

# Logistic Regression for Parkinson's dataset
log_reg_p = LogisticRegression(max_iter=200)
log_reg_p.fit(X_train_p, y_train_p)
y_pred_lr_p = log_reg_p.predict(X_test_p)

print("\nAccuracy (Logistic Regression - Parkinson's):", accuracy_score(y_test_p, y_pred_lr_p))
print("\nClassification Report (Logistic Regression - Parkinson's):\n", classification_report(y_test_p, y_pred_lr_p))
plt.figure()
sns.heatmap(confusion_matrix(y_test_p, y_pred_lr_p), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression (Parkinson's)")
plt.show()

# Random Forest for Parkinson's dataset
rf_p = RandomForestClassifier(n_estimators=100, random_state=42)
rf_p.fit(X_train_p, y_train_p)
y_pred_rf_p = rf_p.predict(X_test_p)

print("\nAccuracy (Random Forest - Parkinson's):", accuracy_score(y_test_p, y_pred_rf_p))
print("\nClassification Report (Random Forest - Parkinson's):\n", classification_report(y_test_p, y_pred_rf_p))
plt.figure()
sns.heatmap(confusion_matrix(y_test_p, y_pred_rf_p), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest (Parkinson's)")
plt.show()
