import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final marks using study hours, attendance, and previous marks.")

data = pd.DataFrame({
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95],
    "previous_marks": [50, 55, 60, 65, 70, 75, 80, 85, 90],
    "final_marks": [52, 57, 62, 67, 72, 77, 82, 87, 92]
})

X = data[["study_hours", "attendance", "previous_marks"]]
y = data["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5
)
attendance = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0
)
previous_marks = st.number_input(
    "Previous Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0
)

if st.button("Predict Final Marks"):
    student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_marks": [previous_marks]
    })
    prediction = model.predict(student)[0]
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Final Marks: {prediction:.2f} / 100")

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
