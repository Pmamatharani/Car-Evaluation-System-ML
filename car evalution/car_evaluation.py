
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
print("Loading Car Evaluation Dataset...")

data = fetch_openml(name="car", version=1, as_frame=True)
df = data.frame

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Separate input and output
X = df.drop(columns=["class"])
y = df["class"]

# Encode categorical data
X = pd.get_dummies(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a sample car
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Car Prediction:")
print("Car Evaluation:", encoder.inverse_transform(prediction)[0])
