import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
# Load Boston Housing dataset from external URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Process the data format
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # Horizontally stack
target = raw_df.values[1::2, 2]
X = pd.DataFrame(data)
y = pd.DataFrame(target, columns=["PRICE"])

# Normalize numerical features
X = (X - X.mean()) / X.std()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Pandas data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Step 2: Define the Linear Regression Model in PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer with one output

    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]  # Number of features
model = LinearRegressionModel(input_dim)

# Step 3: Training the Model
# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 4: Evaluating the Model
# Switch to evaluation mode and make predictions
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    # Convert predictions to NumPy arrays for metric calculations
    train_preds_np = train_preds.numpy()
    test_preds_np = test_preds.numpy()

    # Calculate metrics for training data
    train_rmse = mean_squared_error(y_train, train_preds_np, squared=False)  # RMSE
    train_mae = mean_absolute_error(y_train, train_preds_np)  # MAE

    # Calculate metrics for test data
    test_rmse = mean_squared_error(y_test, test_preds_np, squared=False)  # RMSE
    test_mae = mean_absolute_error(y_test, test_preds_np)  # MAE

# Print results
print(f'Training RMSE: {train_rmse:.4f}')
print(f'Training MAE: {train_mae:.4f}')
print(f'Testing RMSE: {test_rmse:.4f}')
print(f'Testing MAE: {test_mae:.4f}')

# Visualization
plt.figure(figsize=(10, 5))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, test_preds.numpy(), alpha=0.7, color='red', label='Predicted vs Actual')

# Fit a line to the data points (using polyfit for a linear regression line)
slope, intercept = np.polyfit(y_test.values.flatten(), test_preds.numpy().flatten(), 1)
fit_line = slope * y_test.values.flatten() + intercept

# Plot the fitting line (blue color)
plt.plot(y_test, fit_line, color='blue', label='Fitting Line')

# Add labels and title
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices on Test Set with Fitting Line")

# Display the legend
plt.legend()

# Show the plot
plt.show()
