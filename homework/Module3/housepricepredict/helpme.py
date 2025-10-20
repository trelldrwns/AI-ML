import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'House': [1, 2, 3, 4, 5],
    'Area': [1200, 1400, 1600, 1700, 1850],
    'Rooms': [3, 4, 3, 5, 4],
    'Distance': [5, 3, 8, 2, 4],
    'Age': [10, 3, 20, 15, 7],
    'Price': [120, 150, 130, 180, 170]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)
print("\n" + "="*50 + "\n")

# Prepare features and target variable
X = df[['Area', 'Rooms', 'Distance', 'Age']]
y = df['Price']

# Since we have limited data, we'll use all of it for training
# In a real scenario, we'd split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training data
y_pred = model.predict(X)

print("Model Performance:")
print("=" * 30)

# Model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {model.coef_[i]:.4f}")

# Calculate metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\nModel Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Create a results dataframe
results_df = df.copy()
results_df['Predicted_Price'] = y_pred
results_df['Error'] = results_df['Price'] - results_df['Predicted_Price']

print(f"\n{'='*50}")
print("PREDICTION RESULTS:")
print(f"{'='*50}")
print(results_df.round(2))

# Function to predict price for new houses
def predict_house_price(area, rooms, distance, age):
    features = np.array([[area, rooms, distance, age]])
    prediction = model.predict(features)
    return prediction[0]

print(f"\n{'='*50}")
print("SAMPLE PREDICTIONS:")
print(f"{'='*50}")

# Sample predictions
sample_houses = [
    (1500, 3, 4, 8),   # Similar to existing data
    (2000, 4, 6, 5),   # Larger area, moderate distance
    (1000, 2, 10, 12)  # Smaller house, far away
]

print("\nNew House Predictions:")
for i, (area, rooms, distance, age) in enumerate(sample_houses, 1):
    price = predict_house_price(area, rooms, distance, age)
    print(f"House {i}: Area={area}, Rooms={rooms}, Distance={distance}km, Age={age}yrs → Predicted Price: {price:.2f} Lacs")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted Prices
plt.subplot(2, 3, 1)
plt.scatter(results_df['House'], results_df['Price'], color='blue', label='Actual', alpha=0.7)
plt.scatter(results_df['House'], results_df['Predicted_Price'], color='red', label='Predicted', alpha=0.7)
plt.plot(results_df['House'], results_df['Price'], 'b--', alpha=0.3)
plt.plot(results_df['House'], results_df['Predicted_Price'], 'r--', alpha=0.3)
plt.xlabel('House')
plt.ylabel('Price (Lacs)')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Area vs Price
plt.subplot(2, 3, 2)
plt.scatter(df['Area'], df['Price'], color='green')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (Lacs)')
plt.title('Area vs Price')
plt.grid(True, alpha=0.3)

# Plot 3: Rooms vs Price
plt.subplot(2, 3, 3)
plt.scatter(df['Rooms'], df['Price'], color='orange')
plt.xlabel('Rooms')
plt.ylabel('Price (Lacs)')
plt.title('Rooms vs Price')
plt.grid(True, alpha=0.3)

# Plot 4: Distance vs Price
plt.subplot(2, 3, 4)
plt.scatter(df['Distance'], df['Price'], color='purple')
plt.xlabel('Distance (km)')
plt.ylabel('Price (Lacs)')
plt.title('Distance vs Price')
plt.grid(True, alpha=0.3)

# Plot 5: Age vs Price
plt.subplot(2, 3, 5)
plt.scatter(df['Age'], df['Price'], color='brown')
plt.xlabel('Age (years)')
plt.ylabel('Price (Lacs)')
plt.title('Age vs Price')
plt.grid(True, alpha=0.3)

# Plot 6: Prediction Errors
plt.subplot(2, 3, 6)
plt.bar(results_df['House'], results_df['Error'], color='gray', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='-', alpha=0.8)
plt.xlabel('House')
plt.ylabel('Prediction Error')
plt.title('Prediction Errors')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance analysis
print(f"\n{'='*50}")
print("FEATURE ANALYSIS:")
print(f"{'='*50}")

# Calculate correlation matrix
correlation_matrix = df[['Area', 'Rooms', 'Distance', 'Age', 'Price']].corr()
print("\nCorrelation with Price:")
print(correlation_matrix['Price'].sort_values(ascending=False))