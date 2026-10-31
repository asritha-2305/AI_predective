import streamlit as st
import pandas as pd
import joblib

# Load trained AI model
model = joblib.load("motor_model.pkl")

# Dashboard title
st.title("AI-Based Predictive Maintenance System")

st.write("Monitor motor health using AI")

st.subheader("Enter Motor Sensor Readings")

# Sensor inputs
temperature = st.slider(
    "Temperature (°C)", 30, 110, 60
)

vibration = st.slider(
    "Vibration (mm/s)", 0.1, 15.0, 2.5
)

current = st.slider(
    "Current (A)", 1, 30, 12
)

# Prediction button
if st.button("Predict Machine Condition"):

    input_data = pd.DataFrame(
        [[temperature, vibration, current]],
        columns=["Temperature", "Vibration", "Current"]
    )

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == "Normal":
        st.success("Machine Condition: NORMAL")

    elif prediction[0] == "Warning":
        st.warning("Machine Condition: WARNING")

    else:
        st.error("Machine Condition: CRITICAL")

    st.write("AI Prediction:", prediction[0])

st.info("This is an educational simulation using synthetic sensor data.")
# Sensor Data Visualization

st.subheader("Motor Sensor Data Analysis")

data = pd.read_csv("motor_data.csv")

st.metric("Total Sensor Records", len(data))

st.write("Temperature Analysis")
st.line_chart(data[["Temperature"]].head(100))

st.write("Vibration Analysis")
st.line_chart(data[["Vibration"]].head(100))

st.write("Current Analysis")
st.line_chart(data[["Current"]].head(100))

st.write("Machine Condition Distribution")
st.bar_chart(data["Condition"].value_counts())