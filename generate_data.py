import pandas as pd
import numpy as np

# Generate sample motor sensor readings
np.random.seed(42)

n = 3000

temperature = np.random.uniform(30, 110, n)
vibration = np.random.uniform(0.1, 15, n)
current = np.random.uniform(1, 30, n)

# Assign machine condition
def predict_condition(temp, vib, curr):

    if temp >= 90 or vib >= 10 or curr >= 25:
        return "Critical"

    elif temp >= 75 or vib >= 6 or curr >= 20:
        return "Warning"

    else:
        return "Normal"

condition = []

for i in range(n):
    result = predict_condition(
        temperature[i],
        vibration[i],
        current[i]
    )
    condition.append(result)

# Create dataset
data = pd.DataFrame({
    "Temperature": temperature,
    "Vibration": vibration,
    "Current": current,
    "Condition": condition
})

# Save dataset
data.to_csv("motor_data.csv", index=False)

print("Motor sensor dataset created successfully!")
print(data.head())