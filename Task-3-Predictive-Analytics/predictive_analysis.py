import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("dataset/submission.csv")

# Display first rows
print("Dataset Preview:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Features and target
X = data[['id']]
y = data['sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Evaluate accuracy
mae = mean_absolute_error(y, predictions)

print("\nMean Absolute Error:", mae)

# Predict future sales
future_ids = pd.DataFrame({
    'id': [3001000, 3002000, 3003000]
})

future_predictions = model.predict(future_ids)

print("\nFuture Predictions:")
for i, pred in zip(future_ids['id'], future_predictions):
    print(f"ID {i}: Predicted Sales = {pred:.2f}")

# Visualization
plt.figure(figsize=(10, 6))

# Actual data
plt.scatter(X, y, color='blue', label='Actual Sales')

# Regression line
plt.plot(X, predictions, color='red', label='Regression Line')

# Future predictions
plt.scatter(
    future_ids,
    future_predictions,
    color='green',
    label='Future Predictions',
    s=100
)

plt.xlabel("ID")
plt.ylabel("Sales")
plt.title("Predictive Analytics Using Historical Sales Data")
plt.legend()

plt.show()
