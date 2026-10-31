import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load sensor dataset
data = pd.read_csv("motor_data.csv")

# Select input features
X = data[["Temperature", "Vibration", "Current"]]

# Select output
y = data["Condition"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("AI Model Training Completed!")
print("Model Accuracy:", accuracy * 100, "%")

print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "motor_model.pkl")

print("AI Model Saved Successfully!")