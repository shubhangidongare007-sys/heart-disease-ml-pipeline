import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# =======================================================
# # Q1. (Data Loading & Preprocessing)
# =======================================================
# Load the dataset and automatically detect the delimiter (comma, tab, etc.)
df = pd.read_csv('heart.csv', sep=None, engine='python')

# If auto-detection fails and reads everything as 1 column, reload using a semicolon delimiter
if df.shape[1] <= 1:
    df = pd.read_csv('heart.csv', sep=';')

# Convert text/string columns safely into categorical integer codes (e.g., Male/Female to 1/0)
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        df[col] = df[col].astype('category').cat.codes

# Drop any rows containing missing or null values
df = df.dropna()

# Separate independent variables/features (X) from the dependent target label (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize StandardScaler to normalize feature values to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target values into a clean 64-bit integer numpy array
y_clean = np.array(y, dtype=np.int64)

# =======================================================
# # Q2. (Train-Test Split)
# =======================================================
# Split data into 80% training set and 20% testing set with a fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_clean, test_size=0.20, random_state=42
)

# Print the shape configurations to verify data allocation sizes
print("--- Data Split Verification ---")
print(f"X_train shape: {X_train.shape} (Rows, Features)")
print(f"X_test shape:  {X_test.shape} (Rows, Features)")

# =======================================================
# # Q3. (Building Logistic Regression Model)
# =======================================================
# Initialize the Logistic Regression model with 1000 iterations max limit for convergence
model = LogisticRegression(max_iter=1000)

# Fit and train the model using the training datasets
model.fit(X_train, y_train)

# =======================================================
# # Q4. (Making Predictions)
# =======================================================
# Predict classifications for the unseen test feature dataset
y_pred = model.predict(X_test)

# Print and verify the first 10 actual target entries against the predicted values side-by-side
print("\n--- Q4: Predictions Verification ---")
print("First 10 Actual values (y_test):   ", y_test[:10])
print("First 10 Predicted values (y_pred):", y_pred[:10])

# =======================================================
# # Q5. (Confusion Matrix)
# =======================================================
# Generate the raw data classification confusion matrix metrics array
cm = confusion_matrix(y_test, y_pred)
print("\n--- Q5: Confusion Matrix ---")
print(cm)

# Configure and plot a professional colored heatmap for the calculated confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Heart Disease'], 
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()  # Display the graphical window popup onto the screen

# =======================================================
# # Q6. (Model Evaluation Metrics)
# =======================================================
# Calculate and print explicit mathematical validation performance metrics
print("\n--- Q6: Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Print out a detailed structured summary classification breakdown report
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# =======================================================
# # Q7. (Saving the Model Pipeline Files)
# =======================================================
# Save the model object, standard scaler object, and column lists as standalone .pkl files
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("\nModel pipeline files saved successfully!")