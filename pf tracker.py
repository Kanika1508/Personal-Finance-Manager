import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    'Hours': [1.5, 2.3, 3.1, 4.0, 5.2, 6.1, 7.4, 8.5, 9.6, 10.0],
    'Scores': [20, 30, 35, 40, 55, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

hours_studied = 7.0
predicted_score = model.predict([[hours_studied]])
print(f"Predicted score for {hours_studied} hours of study: {predicted_score[0]:.2f}")