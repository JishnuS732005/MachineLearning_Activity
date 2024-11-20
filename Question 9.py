"""9. Real Estate Price Prediction  
Scenario: A real estate company wants to predict house prices based on features like location, size, and number of bedrooms.  
Question: If your model's R-squared value is 0.85, what does this indicate about the model's performance? Are there any limitations
          to using R-squared as the sole metric?"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Example Dataset
data = {
    "Location_Score": [8, 6, 9, 7, 6, 5, 8],
    "Size_SqFt": [1200, 850, 1500, 1000, 800, 650, 1300],
    "Bedrooms": [3, 2, 4, 3, 2, 1, 3],
    "Price": [300000, 200000, 400000, 250000, 180000, 150000, 320000],
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Location_Score", "Size_SqFt", "Bedrooms"]]
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# R-squared and MSE
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r_squared:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Coefficients Interpretation
coefficients = model.coef_
intercept = model.intercept_
feature_names = X.columns

print("Intercept:", intercept)
print("Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
