import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample student dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [60, 65, 70, 72, 75, 80, 85, 88, 92, 95],
    "previous_marks": [45, 50, 55, 60, 62, 68, 72, 78, 84, 90],
    "final_marks": [48, 52, 57, 61, 65, 70, 75, 80, 87, 93]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Input features
X = df[["study_hours", "attendance", "previous_marks"]]

# Target
y = df["final_marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("\nActual Marks:", list(y_test))
print("Predicted Marks:", [round(x, 2) for x in predictions])

# Calculate error
error = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", round(error, 2))

# Predict a new student's performance
new_student = [[7, 85, 75]]

predicted_marks = model.predict(new_student)

print(
    "\nPredicted final marks for new student:",
    round(predicted_marks[0], 2)
)